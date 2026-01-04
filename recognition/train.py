import os
import argparse
import torch
from torch.utils.data import DataLoader

from recognition.dataset import RecConfig, QuickDrawNpzDataset
from recognition.model import SmallCNN


def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="data/QuickDraw_generation")
    ap.add_argument("--classes", type=str, required=True, help='comma-separated, e.g. "airplane,cat,fish,umbrella"')
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--image_size", type=int, default=96)
    ap.add_argument("--out_dir", type=str, default="recognition/results")
    ap.add_argument("--max_samples", type=int, default=None, help="limit samples per class")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Updated to use data_dir and correct RecConfig
    train_ds = QuickDrawNpzDataset(RecConfig(data_dir=args.data_root, classes=classes, split="train", image_size=args.image_size, max_samples=args.max_samples))
    val_ds   = QuickDrawNpzDataset(RecConfig(data_dir=args.data_root, classes=classes, split="val", image_size=args.image_size, max_samples=args.max_samples))

    # num_workers=0 to avoid Windows pickle issues
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, persistent_workers=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, persistent_workers=False)

    model = SmallCNN(num_classes=len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.CrossEntropyLoss()

    best_acc = -1.0
    from tqdm import tqdm

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_acc = 0.0
        pbar = tqdm(train_loader, desc=f"train {ep}/{args.epochs}")
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            logits = model(x)
            loss = crit(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            tr_acc += accuracy(logits.detach(), y.detach())
            pbar.set_postfix({"loss": loss.item()})

        tr_loss /= len(train_loader)
        tr_acc  /= len(train_loader)

        model.eval()
        va_loss = 0.0
        va_acc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                logits = model(x)
                loss = crit(logits, y)
                va_loss += loss.item()
                va_acc += accuracy(logits, y)

        va_loss /= len(val_loader)
        va_acc  /= len(val_loader)

        print(f"[ep {ep}] train loss={tr_loss:.4f} acc={tr_acc:.4f} | val loss={va_loss:.4f} acc={va_acc:.4f}")

        # save last
        torch.save({"model": model.state_dict(), "classes": classes}, os.path.join(args.out_dir, "last.pt"))

        # save best
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "classes": classes}, os.path.join(args.out_dir, "best.pt"))

    print("best val acc:", best_acc)


if __name__ == "__main__":
    main()
