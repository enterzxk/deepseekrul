import json, sys, platform, subprocess, hashlib
from pathlib import Path
import torch, numpy, pandas, sklearn
out = dict(
    python=sys.version,
    executable=sys.executable,
    platform=platform.platform(),
    torch=torch.__version__,
    cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    numpy=numpy.__version__, pandas=pandas.__version__, sklearn=sklearn.__version__,
    nvidia=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip(),
)
Path('/home/user/rulb/tim_20261105/full_20261105/runtime.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
