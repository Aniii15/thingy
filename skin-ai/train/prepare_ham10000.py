import argparse
import shutil
from pathlib import Path
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle(dest: Path):
    api = KaggleApi()
    api.authenticate()
    dest.mkdir(parents=True, exist_ok=True)
    print("Downloading HAM10000 from Kaggle (via Kaggle API)...")
    # This downloads a single zip and unzips it into dest/skin-cancer-mnist-ham10000
    api.dataset_download_files(
        "kmader/skin-cancer-mnist-ham10000",
        path=str(dest),
        unzip=True
    )
    return dest / "skin-cancer-mnist-ham10000"


def extract_zips(src_dir: Path, out_images_dir: Path):
    out_images_dir.mkdir(parents=True, exist_ok=True)
    parts = ["HAM10000_images_part_1.zip", "HAM10000_images_part_2.zip"]
    for p in parts:
        z = src_dir / p
        if not z.exists():
            raise FileNotFoundError(
                f"Missing {z}. Kaggle download may have failed.")
        print(f"Extracting {z} ...")
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(out_images_dir)


def build_dataset(raw_dir: Path, out_dir: Path, val_ratio: float = 0.2, limit_per_class: int = 0):
    meta_csv = raw_dir / "HAM10000_metadata.csv"
    images_dir = raw_dir / "images_extracted"
    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing {meta_csv}")
    if not images_dir.exists():
        extract_zips(raw_dir, images_dir)

    df = pd.read_csv(meta_csv)
    classes = sorted(df["dx"].unique())
    print("Found classes:", classes)

    if limit_per_class and limit_per_class > 0:
        df = df.groupby("dx", group_keys=False).apply(
            lambda x: x.sample(min(len(x), limit_per_class), random_state=42)
        )

    df["img_path"] = df["image_id"].apply(lambda x: images_dir / f"{x}.jpg")
    df = df[df["img_path"].apply(lambda p: p.exists())].reset_index(drop=True)

    train_df, val_df = train_test_split(
        df, test_size=val_ratio, random_state=42, stratify=df["dx"]
    )

    for split_name, split_df in [("train", train_df), ("val", val_df)]:
        for cls in classes:
            (out_dir / split_name / cls).mkdir(parents=True, exist_ok=True)
        print(f"Copying {split_name} images ...")
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            src = row["img_path"]
            cls = row["dx"]
            dst = out_dir / split_name / cls / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

    print(f"Prepared dataset at: {out_dir}")
    print("Class folders in train/:",
          [p.name for p in (out_dir / 'train').iterdir() if p.is_dir()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download_dir", type=Path,
                    default=Path("train/ham10000_raw"))
    ap.add_argument("--out_dir", type=Path, default=Path("train/data"))
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--limit_per_class", type=int, default=0,
                    help="Limit images per class (e.g., 150) for quick run")
    args = ap.parse_args()

    args.download_dir.mkdir(parents=True, exist_ok=True)
    kaggle_dir = args.download_dir / "skin-cancer-mnist-ham10000"
    if kaggle_dir.exists():
        print("Found existing download, skipping download.")
    else:
        kaggle_dir = download_kaggle(args.download_dir)

    build_dataset(kaggle_dir, args.out_dir, val_ratio=args.val_ratio,
                  limit_per_class=args.limit_per_class)


if __name__ == "__main__":
    main()
