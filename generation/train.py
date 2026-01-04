# -*- coding: utf-8 -*-
import os
import argparse
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GenConfig, QuickDrawGenerationDataset, collate_batch
from model import ModelConfig, SketchVAE, loss_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/QuickDraw_generation")
    ap.add_argument("--classes", type=str, default="", help="comma-separated, e.g. cat,dog,airplane")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--out_dir", type=str, default="generation/results")
    ap.add_argument("--max_samples", type=int, default=None, help="limit samples per class")
    ap.add_argument("--num_workers", type=int, default=2, help="dataloader workers")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()] or None

    train_ds = QuickDrawGenerationDataset(GenConfig(data_dir=args.data_dir, classes=classes, split="train", max_len=args.max_len, normalize=False, max_samples=args.max_samples))
    valid_ds = QuickDrawGenerationDataset(GenConfig(data_dir=args.data_dir, classes=classes, split="valid", max_len=args.max_len, normalize=False, max_samples=args.max_samples))


    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reduced model size for CPU efficiency: enc 256->128, dec 512->256
    cfg = ModelConfig(num_classes=len(train_ds.class_names), max_len=args.max_len, enc_hidden=128, dec_hidden=256)
    model = SketchVAE(cfg).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = 1e18
    for ep in range(1, args.epochs + 1):
        # train
        model.train()
        tr = {"total": 0.0, "rec": 0.0, "pen": 0.0, "kl": 0.0}
        n = 0

        for batch in tqdm(train_loader, desc=f"train {ep}/{args.epochs}"):
            seq = batch["seq"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device)

            out = model(seq, mask, label)
            losses = loss_fn(out, seq, mask, beta=args.beta)

            optim.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            bs = seq.size(0)
            n += bs
            for k in tr:
                tr[k] += float(losses[k].item()) * bs

        for k in tr:
            tr[k] /= max(n, 1)

        # valid
        model.eval()
        va = {"total": 0.0, "rec": 0.0, "pen": 0.0, "kl": 0.0}
        m = 0
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="valid", leave=False):
                seq = batch["seq"].to(device)
                mask = batch["mask"].to(device)
                label = batch["label"].to(device)

                out = model(seq, mask, label)
                losses = loss_fn(out, seq, mask, beta=args.beta)

                bs = seq.size(0)
                m += bs
                for k in va:
                    va[k] += float(losses[k].item()) * bs

        for k in va:
            va[k] /= max(m, 1)

        print(f"[ep {ep}] train={tr} valid={va}")

        ckpt = {"model": model.state_dict(), "cfg": asdict(cfg), "class_names": train_ds.class_names}
        torch.save(ckpt, os.path.join(args.out_dir, "last.pt"))
        if va["total"] < best:
            best = va["total"]
            torch.save(ckpt, os.path.join(args.out_dir, "best.pt"))
            print("  saved best.pt")


if __name__ == "__main__":
    main()
