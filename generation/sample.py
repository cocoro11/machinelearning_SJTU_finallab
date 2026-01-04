# -*- coding: utf-8 -*-
import os
import argparse
import torch

from model import SketchVAE, ModelConfig
from render import plot_stroke


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="generation/results/best.pt")
    ap.add_argument("--out_dir", type=str, default="generation/samples")
    ap.add_argument("--class_name", type=str, required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    class_names = ckpt["class_names"]
    if args.class_name not in class_names:
        raise ValueError(f"class_name must be one of: {class_names}")

    cid = class_names.index(args.class_name)
    cfg = ModelConfig(**ckpt["cfg"])
    model = SketchVAE(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        z = torch.randn((args.n, cfg.z_dim), device=device)
        y = torch.full((args.n,), cid, dtype=torch.long, device=device)
        seq = model.decoder.sample(z, y, max_len=args.max_len, temperature=args.temperature)

    seq = seq.detach().cpu().numpy()
    for i in range(args.n):
        out_png = os.path.join(args.out_dir, f"{args.class_name}_{i}_t{args.temperature}.png")
        plot_stroke(seq[i], save_path=out_png, title=f"{args.class_name} (t={args.temperature})")
        print("saved", out_png)


if __name__ == "__main__":
    main()
