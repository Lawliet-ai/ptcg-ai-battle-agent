"""Export the trained policy head to .npz for the numpy runtime plug."""
import sys, os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_policy import PolicyNet

pt = sys.argv[1] if len(sys.argv) > 1 else "train/policy_net.pt"
out = sys.argv[2] if len(sys.argv) > 2 else "agent/policy_net.npz"

ckpt = torch.load(pt, map_location="cpu")
m = PolicyNet(d=ckpt["emb.weight"].shape[1])
m.load_state_dict(ckpt)
sd = {k: v.numpy().astype(np.float32) for k, v in m.state_dict().items()}
np.savez_compressed(out,
                    emb=sd["emb.weight"], sfw=sd["sfc.weight"], sfb=sd["sfc.bias"],
                    card=sd["card.weight"], atk=sd["atk.weight"], otype=sd["otype.weight"],
                    h1w=sd["h1.weight"], h1b=sd["h1.bias"],
                    ow=sd["out.weight"], ob=sd["out.bias"])
print("exported", out, f"({os.path.getsize(out)/1e6:.1f} MB)")
