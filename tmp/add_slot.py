from pathlib import Path
p=Path('src/candidates/dspt.py'); s=p.read_text()

# 1) add slot scorer in __init__
old="        self.sensor_score = nn.Linear(d_model, 1)"
new=('        self.sensor_score = nn.Linear(d_model, 1)\n'
     '        # slot pooling: 1 global + 4 contiguous segments of the valid prefix\n'
     '        self.slot_score = nn.Linear(d_model, 1)\n'
     '        self.n_slots = 5')
assert old in s, 'init anchor'
s=s.replace(old,new,1)

# 2) add slot pooling method
old2="    def forward(self, x, mask=None, baseline=None, cycle=None):"
new2=('    def slot_pool(self, h, mask):\n'
      '        """global mean + 4 segment means over the valid prefix, softmax over 5 slots."""\n'
      '        B, L, D = h.shape\n'
      '        valid = mask.bool()\n'
      '        m = valid.sum(dim=1).clamp_min(1)\n'
      '        slots = [masked_mean(h, mask)]\n'
      '        # segment index counted from the END so slot 4 is the most recent quarter\n'
      '        idx = torch.arange(L, device=h.device).unsqueeze(0).expand(B, L)\n'
      '        pos_from_end = (m.unsqueeze(1) - 1) - (idx - (L - m.unsqueeze(1)))\n'
      '        for k in range(4):\n'
      '            lo = (pos_from_end * 4 // m.unsqueeze(1).clamp_min(1))\n'
      '            seg = valid & (lo == k)\n'
      '            slots.append(masked_mean(h, seg.to(h.dtype)))\n'
      '        S = torch.stack(slots, dim=1)                       # (B,5,D)\n'
      '        w = torch.softmax(self.slot_score(S).squeeze(-1), dim=1)  # (B,5)\n'
      '        return (w.unsqueeze(-1) * S).sum(dim=1), w\n'
      '\n'
      '    def forward(self, x, mask=None, baseline=None, cycle=None, mode="dual"):')
assert old2 in s, 'fwd anchor'
s=s.replace(old2,new2,1)

# 3) use slot pooling when mode == 'slot'
old3="        y = self.reg_head(h_f)"
new3=('        if mode == "slot":\n'
      '            h_s, w_slots = self.slot_pool(h, mask)\n'
      '            y = self.reg_head(h_s)\n'
      '            return {"prediction": y, "y_A": self.reg_head(h_A), "y_B": self.reg_head(h_B),\n'
      '                    "gate": g, "slot_w": w_slots, "embedding": h_s}\n'
      '        y = self.reg_head(h_f)')
assert old3 in s, 'pred anchor'
s=s.replace(old3,new3,1)
p.write_text(s); print('added slot pooling')