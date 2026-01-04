
# -*- coding: utf-8 -*-
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

try:
    # Import Task B (Generation) modules
    from generation.model import SketchVAE, ModelConfig
    from generation.dataset import GenConfig

    # Import Task A (Recognition) modules
    from recognition.model import SmallCNN
    from recognition.dataset import stroke3_to_image # Reuse your optimized rendering logic
except Exception as e:
    print(f"Import Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_ckpt", type=str, default="generation/results/best.pt")
    ap.add_argument("--rec_ckpt", type=str, default="recognition/results/best.pt")
    ap.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate per class")
    ap.add_argument("--max_len", type=int, default=200)
    ap.add_argument("--image_size", type=int, default=96)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running pipeline evaluation on: {device}", flush=True)

    # 1. Load Generators
    if not os.path.exists(args.gen_ckpt):
        print(f"Error: Generation checkpoint not found at {args.gen_ckpt}", flush=True)
        # Attempt to see if we have ANY result
        return
    
    print("Loading Generation Model...", flush=True)
    gen_ckpt_data = torch.load(args.gen_ckpt, map_location=device)
    # Handle cfg mismatch if saved structure differs
    if "cfg" in gen_ckpt_data:
        # Reconstruct GenConfig from dict
        gen_cfg_dict = gen_ckpt_data["cfg"]
        # Filter keys that current ModelConfig accepts (in case of version diff)
        # Assuming direct match for now
        gen_cfg = ModelConfig(**gen_cfg_dict)
    else:
        # Fallback default
        print("Warning: Config not found in ckpt, using defaults")
        gen_cfg = ModelConfig(num_classes=len(gen_ckpt_data["class_names"]))

    gen_model = SketchVAE(gen_cfg).to(device)
    gen_model.load_state_dict(gen_ckpt_data["model"])
    gen_model.eval()
    
    gen_class_names = gen_ckpt_data["class_names"]
    print(f"Generator classes: {gen_class_names}")

    # 2. Load Recognizer
    if not os.path.exists(args.rec_ckpt):
        print(f"Error: Recognition checkpoint not found at {args.rec_ckpt}")
        print("Skipping recognition step, generating sketches only.")
        has_rec = False
    else:
        has_rec = True
        print("Loading Recognition Model...")
        rec_ckpt_data = torch.load(args.rec_ckpt, map_location=device)
        rec_classes = rec_ckpt_data["classes"] # Expected list of class names
        rec_model = SmallCNN(num_classes=len(rec_classes)).to(device)
        rec_model.load_state_dict(rec_ckpt_data["model"])
        rec_model.eval()
        print(f"Recognizer classes: {rec_classes}")

        # Map gen class ID -> rec class ID
        # They might be different if training sets differed order
        gen2rec = {}
        for idx, name in enumerate(gen_class_names):
            if name in rec_classes:
                gen2rec[idx] = rec_classes.index(name)
            else:
                print(f"Warning: Generator class '{name}' not in Recognizer classes. Ignoring.")
                gen2rec[idx] = -1

    # 3. Evaluate Loop
    total = 0
    correct = 0
    
    # Pre-define transform (same as in your optimized dataset)
    transform = lambda img: torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)

    print(f"\nGenerating {args.n_samples} samples per class and evaluating...")
    
    with torch.no_grad():
        for gen_cid, name in enumerate(gen_class_names):
            if has_rec and gen2rec[gen_cid] == -1:
                continue
                
            rec_target = gen2rec[gen_cid] if has_rec else -1
            
            # Batch generation
            # Generate z
            z = torch.randn((args.n_samples, gen_cfg.z_dim), device=device)
            y = torch.full((args.n_samples,), gen_cid, dtype=torch.long, device=device)
            
            # Sample from VAE
            # returns (B, L, 5)
            seqs = gen_model.decoder.sample(z, y, max_len=args.max_len, temperature=0.7)
            seqs_cpu = seqs.cpu().numpy() # (N, L, 5)
            
            # Process each sample
            class_correct = 0
            
            for i in range(args.n_samples):
                stroke5 = seqs_cpu[i]
                
                # Convert back to stroke3 (dx, dy, pen_up)
                # stroke5: (dx, dy, p1, p2, p3) -> stroke3: (dx, dy, pen_up)
                # Logic: pen_down(p1) -> 0, pen_up(p2) -> 1, eos(p3) -> 1
                # Usually we just take p2
                
                # Truncate at EOS
                # p3 is index 4
                s = stroke5
                eos_idx = np.where(s[:, 4] > 0.5)[0]
                if len(eos_idx) > 0:
                    s = s[:eos_idx[0]+1]
                
                # Reconstruct (dx, dy, pen)
                # p2 is pen_up probability (or bit)
                # In sample output it is one-hot from multinomial usually, or raw.
                # model sample logic: 
                # step[torch.arange(B), 2 + pen] = 1.0
                # so s[:, 3] is p2 (pen_up)
                
                stroke3 = np.zeros((len(s), 3), dtype=np.float32)
                stroke3[:, 0:2] = s[:, 0:2]
                stroke3[:, 2] = s[:, 3] # p2 column
                
                if has_rec:
                    # Render to image using your optimized PIL logic
                    img = stroke3_to_image(stroke3, image_size=args.image_size)
                    
                    # Transform to tensor
                    x = transform(img).unsqueeze(0).to(device) # (1, 1, 96, 96)
                    
                    # Predict
                    logits = rec_model(x)
                    pred = logits.argmax(dim=1).item()
                    
                    if pred == rec_target:
                        class_correct += 1
                        correct += 1
                    total += 1
            
            if has_rec:
                acc = class_correct / args.n_samples
                print(f"Class: {name:<10} | Acc: {acc:.4f} ({class_correct}/{args.n_samples})")
            else:
                print(f"Class: {name:<10} | Generated {args.n_samples} sketches (No evaluation)")

    if has_rec and total > 0:
        print(f"\nOverall Generation Accuracy: {correct/total:.4f}")
        print("Note: This metric evaluates how recognizable the generated sketches are.")

if __name__ == '__main__':
    main()
