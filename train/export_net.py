"""Export the trained torch value net to a plain .npz the runtime plug can load
(numpy-only forward — no torch in the ladder bundle)."""
import sys, os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_value import ValueNet

pt = sys.argv[1] if len(sys.argv) > 1 else "train/value_net.pt"
out = sys.argv[2] if len(sys.argv) > 2 else "agent/value_net.npz"

ckpt = torch.load(pt, map_location="cpu")
m = ValueNet(d=ckpt["emb.weight"].shape[1])   # infer size from the checkpoint
m.load_state_dict(ckpt)
sd = {k: v.numpy().astype(np.float32) for k, v in m.state_dict().items()}
np.savez_compressed(out,
                    emb=sd["emb.weight"],
                    w1=sd["fc1.weight"], b1=sd["fc1.bias"],
                    w2=sd["fc2.weight"], b2=sd["fc2.bias"],
                    wo=sd["out.weight"], bo=sd["out.bias"])
print("exported", out, f"({os.path.getsize(out)/1e6:.1f} MB)")
