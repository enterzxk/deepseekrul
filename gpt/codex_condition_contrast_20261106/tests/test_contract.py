"""Scientific contracts: causality, nuisance cancellation, pairing, and scoring."""
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tim_contrast.model import ConditionTrend, training_loss, tangent_nasa
from tim_contrast.run import build_model, code_fingerprint
from tim_contrast.data import make_prefixes, nested_prior, save_cache, load_cache, prepare_fold
from tim_contrast.official import infer_checkpoint
from tim_contrast.metrics import engine_statistics, summarize
from tim_contrast.analyze import align_queries, screen, confirmation
from tim_contrast.queue import plan

torch.set_num_threads(2)


def history_fixture(batch=2, sensors=14, length=100):
    torch.manual_seed(4)
    x = torch.randn(batch, length, sensors)
    t = torch.arange(1, length+1).expand(batch, -1).clone()
    c = torch.arange(length).remainder(6).expand(batch, -1).clone()
    mask = torch.ones_like(x, dtype=torch.bool)
    end = torch.full((batch,), length)
    return x, t, c, mask, end


class TrendContracts(unittest.TestCase):
    def test_linear_trend_recovered_despite_condition_offsets(self):
        x, t, c, mask, end = history_fixture()
        rates = torch.linspace(-.02, .03, 14)
        offset = torch.randn(6, 14)*3
        x = t[:, :, None]*rates + offset[c]
        out = ConditionTrend(learned=False)(x, t, c, mask, end)
        expected = torch.tensor([20., 50., 100.])[None, :, None]*rates
        torch.testing.assert_close(out["slope"], expected.expand(2, -1, -1), atol=2e-5, rtol=1e-4)

    def test_condition_offset_invariance_with_nonuniform_learned_weights(self):
        data = history_fixture()
        model = ConditionTrend().eval()
        with torch.no_grad():
            model.weight_net[-1].weight.normal_()
        a = model(*data)
        x, t, c, mask, end = data
        shifted = x + torch.randn(6, 14)[c]*2
        b = model(shifted, t, c, mask, end)
        torch.testing.assert_close(a["stats"], b["stats"], atol=1e-5, rtol=2e-4)
        torch.testing.assert_close(a["weights"], b["weights"], atol=1e-6, rtol=2e-4)

    def test_future_values_cannot_change_trend(self):
        x, t, c, mask, end = history_fixture()
        end.fill_(70)
        model = ConditionTrend().eval()
        a = model(x, t, c, mask, end)["stats"]
        changed = x.clone()
        changed[:, 70:] = 1000*torch.randn_like(changed[:, 70:])
        b = model(changed, t, c, mask, end)["stats"]
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_unobserved_values_and_padding_are_ignored(self):
        x, t, c, mask, end = history_fixture()
        mask[:, :25] = False
        mask[:, 40:50, :4] = False
        model = ConditionTrend().eval()
        a = model(x, t, c, mask, end)["stats"]
        changed = x.clone()
        changed[~mask] = float("nan")
        changed[:, 0, 0] = float("inf")
        b = model(changed, t, c, mask, end)["stats"]
        torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_no_information_is_finite_with_finite_backward(self):
        x, t, c, mask, end = history_fixture()
        x.requires_grad_(True)
        mask.fill_(False)
        out = ConditionTrend()(x, t, c, mask, end)
        self.assertFalse(out["identifiable"].any())
        self.assertTrue(torch.isfinite(out["stats"]).all())
        out["stats"].sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_uniform_initialization_equals_fixed_regression(self):
        data = history_fixture()
        a, b = ConditionTrend()(*data), ConditionTrend(learned=False)(*data)
        torch.testing.assert_close(a["stats"], b["stats"], atol=0, rtol=0)

    def test_unmatched_control_does_not_cancel_condition_offsets(self):
        x, t, c, mask, end = history_fixture()
        c = torch.where(t <= 50, 0, 1)
        model = ConditionTrend(learned=False, match_conditions=False)
        a = model(x, t, c, mask, end)["stats"]
        b = model(x+4*c[:, :, None], t, c, mask, end)["stats"]
        self.assertGreater((a-b).abs().max().item(), 1.)


class ModelContracts(unittest.TestCase):
    def inputs(self):
        x, t, c, mask, end = history_fixture()
        extras = torch.zeros(2, 21)
        extras[:, 14] = .4
        prior = torch.tensor([[.8, 0.], [.9, 0.]])
        return x, t, c, mask, end, extras, prior

    def test_candidate_starts_at_same_pe_prediction(self):
        data = self.inputs()
        base = build_model("pe50", 101, True).eval()
        expected = base(*data)
        for arm in ("fixed_trend", "learned_trend", "unmatched_trend"):
            model = build_model(arm, 101, True).eval()
            torch.testing.assert_close(model(*data), expected, atol=0, rtol=0)

    def test_seed_precedes_construction(self):
        a = build_model("learned_trend", 101, True)
        torch.rand(50)
        b = build_model("learned_trend", 101, True)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key], atol=0, rtol=0)

    def test_all_arms_backward_and_branch_learns_after_zero_start(self):
        data = self.inputs()
        for arm in ("pe50", "pe100", "w100_late", "fixed_trend", "learned_trend", "unmatched_trend"):
            model = build_model(arm, 101, True)
            opt = torch.optim.AdamW(model.parameters(), lr=.001)
            for _ in range(3):
                opt.zero_grad()
                loss = training_loss(model(*data), torch.tensor([40., 80.]), 1.)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                opt.step()
            if arm == "learned_trend":
                self.assertGreater(model.trend.weight_net[-1].weight.grad.abs().sum().item(), 0.)

    def test_tail_loss_is_c1_and_keeps_large_error_gradient(self):
        e = torch.tensor([59.999, 60.001, 120., -120.], requires_grad=True)
        tangent_nasa(e).sum().backward()
        self.assertAlmostEqual(e.grad[0].item(), e.grad[1].item(), places=2)
        self.assertGreater(e.grad[2].item(), 40.)
        self.assertLess(e.grad[3].item(), -30.)
        self.assertTrue(torch.isfinite(e.grad).all())


class DataAndMetricContracts(unittest.TestCase):
    def test_real_preprocessing_and_nested_prior_on_synthetic_train_file(self):
        rng = np.random.default_rng(15)
        rows = []
        for unit in range(1, 21):
            for cycle in range(1, 61+unit):
                rows.append([unit, cycle, 0., 0., 100., *rng.normal(size=21)])
        with TemporaryDirectory() as folder:
            np.savetxt(Path(folder)/"train_FD001.txt", np.array(rows))
            a, b, meta = prepare_fold(folder, "FD001", 0, True)
            self.assertFalse(set(a["unit"]) & set(b["unit"]))
            self.assertEqual(len(set(b["unit"])), 4)
            self.assertTrue(np.isfinite(a["prior"]).all())
            self.assertTrue(np.isfinite(b["prior"]).all())
            self.assertEqual(meta["prior"], "nested_FULL")
            self.assertTrue((b["as_of"] >= 50).all())

    def test_short_official_engine_is_kept_without_reading_target(self):
        frame = pd.DataFrame({"unit": [1, 1, 1], "cycle": [1, 2, 3]})
        x = np.arange(42, dtype=np.float32).reshape(3, 14)/100
        data = make_prefixes(frame, x, np.zeros(3, int), endpoints_only=True, include_targets=False)
        self.assertEqual(len(data["unit"]), 1)
        self.assertEqual(int(data["observed"][0, :, 0].sum()), 3)
        self.assertTrue(np.isnan(data["target"]).all())
        np.testing.assert_array_equal(data["history"][0, -3:], x)
        data["prior"] = np.array([[.8, 0.]], dtype=np.float32)
        with TemporaryDirectory() as folder:
            checkpoint = Path(folder)/"synthetic_test_fixture.pt"
            net = build_model("learned_trend", 101, True)
            torch.save({"state_dict": net.state_dict(), "arm": "learned_trend", "use_prior": True,
                        "epoch": 80, "code_sha256": code_fingerprint()}, checkpoint)
            pred = infer_checkpoint(checkpoint, data, torch.device("cpu"))
            self.assertEqual(pred.shape, (1,))
            self.assertTrue(np.isfinite(pred).all())

    def test_cache_preserves_query_and_observation_masks(self):
        arrays = {"unit": np.array([1, 2]), "target": np.array([10., 20.], np.float32),
                  "observed": np.array([[[True]], [[False]]])}
        with TemporaryDirectory() as folder:
            path = Path(folder)/"cache.npz"
            save_cache(path, arrays, arrays, {"train_sha256": "synthetic"})
            a, b, meta = load_cache(path)
            for k in arrays:
                np.testing.assert_array_equal(a[k], arrays[k])
                np.testing.assert_array_equal(b[k], arrays[k])
            self.assertEqual(meta["train_sha256"], "synthetic")

    def test_extending_or_altering_future_does_not_change_prefix_features(self):
        rng = np.random.default_rng(8)
        frame = pd.DataFrame({"unit": np.ones(140, int), "cycle": np.arange(1, 141)})
        x = rng.normal(size=(140, 14)).astype(np.float32)
        labels = np.arange(140) % 6
        pop = rng.normal(size=(6, 14)).astype(np.float32)
        a = make_prefixes(frame, x, labels, pop)
        changed = x.copy()
        changed[100:] += 100
        b = make_prefixes(frame.iloc[:100], changed[:100], labels[:100], pop)
        for key in ("history", "cycles", "conditions", "observed", "as_of", "extras", "unit"):
            np.testing.assert_array_equal(a[key][:51], b[key])
        self.assertFalse(np.array_equal(a["target"][:51], b["target"]))
        self.assertEqual(int(b["observed"][0, :, 0].sum()), 50)

    def test_inner_prior_fits_exclude_each_held_out_engine(self):
        calls = []
        def fake(frame, subset, train, valid, seed):
            calls.append((set(train), set(valid)))
            return {(int(u), 50): 100. for u in valid}
        with patch("tim_contrast.data.fit_prior", fake):
            nested_prior(pd.DataFrame(), "FD002", np.arange(1, 9), np.arange(9, 11), 0)
        self.assertEqual(len(calls), 5)
        for train, valid in calls:
            self.assertFalse(train & valid)
            self.assertFalse(train & {9, 10})
        self.assertEqual(set.union(*(v for _, v in calls[:4])), set(range(1, 9)))

    def test_engine_equal_metrics_and_exact_nasa(self):
        p = pd.DataFrame({"unit": [1]+[2]*9, "end_cycle": [50]+list(range(50, 59)),
                          "y_true": [0.]*10, "y_pred": [2.]+[4.]*9})
        scores = summarize(engine_statistics(p))
        self.assertAlmostEqual(scores["rmse"], np.sqrt(10.))
        self.assertAlmostEqual(scores["mae"], 3.)
        self.assertAlmostEqual(scores["bias"], 3.)
        self.assertAlmostEqual(scores["nasa"], (np.expm1(.2)+np.expm1(.4))/2)

    def test_duplicates_and_query_dropping_fail_closed(self):
        p = pd.DataFrame({"unit": [1, 2], "end_cycle": [50, 50], "y_true": [1., 2.], "y_pred": [2., 3.]})
        with self.assertRaises(ValueError):
            engine_statistics(pd.concat([p, p]))
        with self.assertRaises(ValueError):
            align_queries(p, p.iloc[:1])
        with self.assertRaises(ValueError):
            align_queries(p, p.assign(y_true=[2., 2.]))


class DecisionContracts(unittest.TestCase):
    def frames(self, tail=False):
        frames = {}
        errors = {"pe50": 3., "pe100": 2., "w100_late": 1.8, "fixed_trend": 1.5, "learned_trend": .5}
        for subset in ("FD002", "FD004"):
            for seed in (101, 202, 303):
                for fold in range(5):
                    units = np.arange(40)+fold*40+1
                    for label, error in errors.items():
                        e = np.full(40, error)
                        if tail:
                            e.fill(20.)
                            if label == "learned_trend":
                                e.fill(0.)
                                e[0] = 70.
                        frames[(subset, label, seed, fold)] = pd.DataFrame({"unit": units,
                             "end_cycle": np.full(40, 100), "y_true": np.full(40, 20.), "y_pred": 20.+e})
        return frames

    def test_screen_confirm_and_queue_use_the_frozen_controls(self):
        with TemporaryDirectory() as folder:
            frames = self.frames()
            report = screen(folder, frames)
            self.assertEqual(report["next"], "CONFIRM")
            jobs = plan("confirm", folder, "full")
            self.assertIn(("FD002", "pe50", 0, 202), jobs)
            self.assertIn(("FD002", "w100_late", 0, 202), jobs)
            self.assertIn(("FD004", "fixed_trend", 4, 303), jobs)
            result = confirmation(folder, frames)
            self.assertEqual(result["status"], "DEVELOPMENT_GATE_PASS")
            self.assertFalse(result["novelty_proven"])
            self.assertFalse(result["submission_ready"])

    def test_rmse_improvement_cannot_override_nasa_failure(self):
        with TemporaryDirectory() as folder:
            report = screen(folder, self.frames(tail=True))
            self.assertEqual(report["next"], "TAIL_SCREEN")
            self.assertFalse((Path(folder)/"analysis"/"promotion.json").exists())


if __name__ == "__main__":
    unittest.main()
