"""
Dataset Split and Indexing Engine.
Generates leakage-free manifest files directly from the dataset archive without
requiring multi-gigabyte in-memory caching.
"""

import os
import sys
import zipfile
import pandas as pd
from typing import List, Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.yolo_parser import parse_yolo_line, id_to_class_name


def index_split(
    zip_obj: zipfile.ZipFile,
    split_name: str,
    root_folder: str = "9 Facial Expressions you need"
) -> List[Dict]:
    """
    Indexes all image and annotation pairs in a given split directly from the zip.
    """
    prefix_img = f"{root_folder}/{split_name}/images/"
    prefix_lbl = f"{root_folder}/{split_name}/labels/"
    
    all_names = set(zip_obj.namelist())
    img_names = [f for f in zip_obj.namelist() if f.startswith(prefix_img) and (f.lower().endswith('.jpg') or f.lower().endswith('.png'))]
    
    records = []
    for img_path in img_names:
        # Match corresponding label file
        filename = img_path.split("/")[-1]
        base_name = filename.rsplit(".", 1)[0]
        lbl_path = f"{prefix_lbl}{base_name}.txt"
        
        if lbl_path not in all_names:
            continue
            
        try:
            lbl_content = zip_obj.read(lbl_path).decode("utf-8").strip()
            if not lbl_content:
                continue
            
            # Read first box (verified 100% single box per file)
            parsed = parse_yolo_line(lbl_content.split("\n")[0])
            if parsed is None:
                continue
                
            cls_id, xc, yc, w, h = parsed
            records.append({
                "image_path": img_path,
                "label_path": lbl_path,
                "class_id": cls_id,
                "class_name": id_to_class_name(cls_id),
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h
            })
        except Exception:
            continue

    return records


def generate_manifests(
    zip_path: str = r"C:\Users\mitra\Downloads\archive.zip",
    output_dir: str = r"d:\Projects\GAN\data\splits",
    pilot_samples_per_class: int = 50,
    seed: int = 42
):
    """
    Generates manifest CSV files for train, valid, test, and a stratified pilot set.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Reading archive: {zip_path}...")
    
    with zipfile.ZipFile(zip_path, "r") as z:
        for split in ["train", "valid", "test"]:
            print(f"Indexing split: {split}...")
            records = index_split(z, split)
            df = pd.DataFrame(records)
            
            out_file = os.path.join(output_dir, f"{split}_manifest.csv")
            df.to_csv(out_file, index=False)
            print(f"Saved {split} manifest: {len(df)} entries -> {out_file}")
            
            # Create a stratified pilot manifest from train set for fast verification
            if split == "train":
                pilot_dfs = []
                for cid in sorted(df["class_id"].unique()):
                    cls_df = df[df["class_id"] == cid]
                    n_take = min(pilot_samples_per_class, len(cls_df))
                    pilot_dfs.append(cls_df.sample(n=n_take, random_state=seed))
                
                pilot_df = pd.concat(pilot_dfs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
                pilot_file = os.path.join(output_dir, "pilot_manifest.csv")
                pilot_df.to_csv(pilot_file, index=False)
                print(f"Saved pilot manifest: {len(pilot_df)} entries ({pilot_samples_per_class}/class) -> {pilot_file}")


if __name__ == "__main__":
    generate_manifests()
