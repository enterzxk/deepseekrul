import sys
import torch

path = sys.argv[1] if len(sys.argv) > 1 else "experiments/punet_matrix/FD001/f0_s42/model.pt"
ck = torch.load(path, map_location="cpu")
print("type:", type(ck).__name__)
if isinstance(ck, dict):
    print("top-level keys:", [k for k in ck.keys()][:10])
    sd = ck.get("model", ck.get("state_dict", ck))
    if isinstance(sd, dict) and all(hasattr(v, "shape") for v in sd.values()):
        print("n params:", len(sd))
        for i, (k, v) in enumerate(sd.items()):
            if i < 20:
                print(f"  {k}: {tuple(v.shape)}")
    else:
        for k, v in list(ck.items())[:10]:
            print(f"  {k}: {type(v).__name__} {getattr(v, 'shape', v) if not isinstance(v, (dict, list)) else ''}")
