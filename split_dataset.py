"""split_dataset.py
Splits a single input folder with class subfolders into train/val/test folders.
Usage: python split_dataset.py --input all_images --out data --train 0.7 --val 0.15 --test 0.15
"""
import argparse
import os
import random
import shutil

def split(input_dir, out_dir, train_ratio, val_ratio, test_ratio, seed=42):
    random.seed(seed)
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist")
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    if not classes:
        raise ValueError(f"No class subfolders found in {input_dir}")
    for split in ('train','val','test'):
        for cls in classes:
            target = os.path.join(out_dir, split, cls)
            os.makedirs(target, exist_ok=True)
    for cls in classes:
        cls_dir = os.path.join(input_dir, cls)
        files = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir,f))]
        random.shuffle(files)
        n = len(files)
        ntrain = int(n * train_ratio)
        nval = int(n * val_ratio)
        ntest = n - ntrain - nval
        train_files = files[:ntrain]
        val_files = files[ntrain:ntrain+nval]
        test_files = files[ntrain+nval:]
        for f in train_files:
            shutil.copy2(os.path.join(cls_dir,f), os.path.join(out_dir,'train',cls,f))
        for f in val_files:
            shutil.copy2(os.path.join(cls_dir,f), os.path.join(out_dir,'val',cls,f))
        for f in test_files:
            shutil.copy2(os.path.join(cls_dir,f), os.path.join(out_dir,'test',cls,f))
    print(f"Split complete. Output written to: {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split dataset into train/val/test folders (copies files).')
    parser.add_argument('--input', required=True, help='Input folder containing class subfolders (e.g., all_images/medical)')
    parser.add_argument('--out', default='data', help='Output folder to create train/val/test (default: data)')
    parser.add_argument('--train', type=float, default=0.7, help='Train split ratio (default 0.7)')
    parser.add_argument('--val', type=float, default=0.15, help='Validation split ratio (default 0.15)')
    parser.add_argument('--test', type=float, default=0.15, help='Test split ratio (default 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    split(args.input, args.out, args.train, args.val, args.test, seed=args.seed)
