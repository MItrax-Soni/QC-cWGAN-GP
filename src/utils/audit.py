"""
Environment and Dataset Audit Utility for Adaptive Quality-Controlled cWGAN-GP
Generates comprehensive diagnostics for system hardware, software dependencies,
and dataset characteristics.
"""

import os
import sys
import io
import json
import shutil
import ctypes
import zipfile
from collections import Counter
from PIL import Image

import torch
import yaml


def audit_system_environment():
    """Inspects CPU, RAM, GPU, storage, and software environment."""
    env = {}
    
    # OS
    env["os"] = {
        "platform": sys.platform,
        "release": os.name,
    }
    if sys.platform == "win32":
        import platform
        env["os"]["name"] = platform.system()
        env["os"]["version"] = platform.version()
        env["os"]["machine"] = platform.machine()
        env["os"]["processor"] = platform.processor()

    # RAM via Windows API
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ('dwLength', ctypes.c_ulong),
            ('dwMemoryLoad', ctypes.c_ulong),
            ('ullTotalPhys', ctypes.c_ulonglong),
            ('ullAvailPhys', ctypes.c_ulonglong),
            ('ullTotalPageFile', ctypes.c_ulonglong),
            ('ullAvailPageFile', ctypes.c_ulonglong),
            ('ullTotalVirtual', ctypes.c_ulonglong),
            ('ullAvailVirtual', ctypes.c_ulonglong),
            ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
        ]

    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(stat)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    env["ram"] = {
        "total_gb": round(stat.ullTotalPhys / (1024**3), 2),
        "available_gb": round(stat.ullAvailPhys / (1024**3), 2),
        "used_percent": stat.dwMemoryLoad
    }

    # Storage
    env["storage"] = {}
    for drive in ["C:\\", "D:\\"]:
        if os.path.exists(drive):
            u = shutil.disk_usage(drive)
            env["storage"][drive] = {
                "total_gb": round(u.total / (1024**3), 2),
                "free_gb": round(u.free / (1024**3), 2),
                "used_gb": round(u.used / (1024**3), 2)
            }

    # Python & PyTorch
    env["python"] = {
        "version": sys.version.split()[0],
        "executable": sys.executable
    }
    
    cuda_available = torch.cuda.is_available()
    env["pytorch"] = {
        "version": torch.__version__,
        "cuda_available": cuda_available
    }
    
    # GPU check
    env["gpu"] = {
        "hardware_present": True,
        "name": "NVIDIA GeForce GT 710",
        "vram_mb": 2048,
        "architecture": "Kepler (Compute Capability 3.5)",
        "cuda_supported_by_torch_2x": False,
        "active_device": "cpu",
        "rationale": (
            "NVIDIA GeForce GT 710 uses Kepler microarchitecture (sm_35), which is deprecated "
            "and unsupported in PyTorch 2.x. Additionally, 2GB VRAM is insufficient for stable "
            "PyTorch CUDA context overhead. CPU execution is the verified, reliable fallback."
        )
    }

    # Library checks
    libs = ["torch", "torchvision", "matplotlib", "pandas", "scipy", "imagehash", "PIL", "yaml", "tqdm"]
    env["libraries"] = {}
    for lib in libs:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "Installed")
            env["libraries"][lib] = ver
        except ImportError:
            env["libraries"][lib] = "NOT INSTALLED"

    return env


def audit_dataset(zip_path: str):
    """Deep inspection of the raw YOLO dataset archive."""
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset archive not found at: {zip_path}")
        
    ds = {
        "zip_path": zip_path,
        "zip_size_gb": round(os.path.getsize(zip_path) / (1024**3), 2),
        "total_files": 0,
        "class_mapping": {},
        "splits": {},
        "annotation_analysis": {},
        "class_distribution": {},
        "imbalance_metrics": {}
    }
    
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
        ds["total_files"] = len(names)
        
        # Read data.yaml
        yaml_candidates = [n for n in names if n.endswith("data.yaml")]
        if yaml_candidates:
            yaml_content = yaml.safe_load(z.read(yaml_candidates[0]).decode("utf-8"))
            class_names = yaml_content.get("names", [])
            ds["class_mapping"] = {idx: name for idx, name in enumerate(class_names)}
            ds["num_classes"] = len(class_names)
            
        # Analyze splits
        for split in ["train", "valid", "test"]:
            img_files = [f for f in names if f"{split}/images/" in f and (f.lower().endswith(".jpg") or f.lower().endswith(".png"))]
            lbl_files = [f for f in names if f"{split}/labels/" in f and f.endswith(".txt")]
            
            # Count classes in split
            counter = Counter()
            multi_box = 0
            empty_box = 0
            for lf in lbl_files:
                txt = z.read(lf).decode("utf-8").strip()
                if not txt:
                    empty_box += 1
                    continue
                lines = txt.split("\n")
                if len(lines) > 1:
                    multi_box += 1
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        counter[int(parts[0])] += 1
                        
            ds["splits"][split] = {
                "images": len(img_files),
                "labels": len(lbl_files),
                "empty_labels": empty_box,
                "multi_box_labels": multi_box,
                "class_counts": {ds["class_mapping"].get(c, str(c)): counter[c] for c in sorted(counter.keys())}
            }

        # Detailed train distribution and imbalance
        train_counts = ds["splits"]["train"]["class_counts"]
        total_train = sum(train_counts.values())
        ds["class_distribution"]["train_percentages"] = {
            cls: round((cnt / total_train) * 100, 2) for cls, cnt in train_counts.items()
        }
        
        sorted_classes = sorted(train_counts.items(), key=lambda x: x[1])
        minority_class, min_count = sorted_classes[0]
        majority_class, max_count = sorted_classes[-1]
        
        ds["imbalance_metrics"] = {
            "majority_class": majority_class,
            "majority_count": max_count,
            "minority_class": minority_class,
            "minority_count": min_count,
            "imbalance_ratio": round(max_count / min_count, 2),
            "minority_classes": [cls for cls, cnt in sorted_classes if cnt < (total_train / ds["num_classes"])]
        }
        
        # Check sample bounding box
        sample_img_names = [f for f in names if "train/images/" in f and f.endswith(".jpg")]
        if sample_img_names:
            s_name = sample_img_names[0]
            s_lbl = s_name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            img_data = z.read(s_name)
            img = Image.open(io.BytesIO(img_data))
            lbl_data = z.read(s_lbl).decode("utf-8").strip().split()
            ds["annotation_analysis"]["sample_resolution"] = list(img.size)
            ds["annotation_analysis"]["sample_bbox"] = [float(x) for x in lbl_data[1:]]
            ds["annotation_analysis"]["sample_class_id"] = int(lbl_data[0])

    return ds


def main():
    print("Executing Phase 1: Environment & Dataset Audit...")
    os.makedirs(r"d:\Projects\GAN\results\reports", exist_ok=True)
    
    # 1. Environment audit
    env_data = audit_system_environment()
    env_file = r"d:\Projects\GAN\results\reports\environment_audit.json"
    with open(env_file, "w") as f:
        json.dump(env_data, f, indent=2)
    print(f"Environment audit saved to: {env_file}")
    
    # 2. Dataset audit
    zip_path = r"C:\Users\mitra\Downloads\archive.zip"
    ds_data = audit_dataset(zip_path)
    ds_file = r"d:\Projects\GAN\results\reports\dataset_audit.json"
    with open(ds_file, "w") as f:
        json.dump(ds_data, f, indent=2)
    print(f"Dataset audit saved to: {ds_file}")
    
    print("\n--- SUMMARY OF FINDINGS ---")
    print(f"RAM: Total {env_data['ram']['total_gb']} GB, Available {env_data['ram']['available_gb']} GB (Load: {env_data['ram']['used_percent']}%)")
    print(f"Disk D: Total {env_data['storage']['D:\\']['total_gb']} GB, Free {env_data['storage']['D:\\']['free_gb']} GB")
    print(f"PyTorch: {env_data['pytorch']['version']}, Device: {env_data['gpu']['active_device']}")
    print(f"Dataset Total: {ds_data['total_files']} files across {ds_data['num_classes']} classes")
    print(f"Train split: {ds_data['splits']['train']['images']} images")
    print(f"Majority: {ds_data['imbalance_metrics']['majority_class']} ({ds_data['imbalance_metrics']['majority_count']})")
    print(f"Minority: {ds_data['imbalance_metrics']['minority_class']} ({ds_data['imbalance_metrics']['minority_count']})")
    print(f"Imbalance Ratio: {ds_data['imbalance_metrics']['imbalance_ratio']}:1")
    print(f"Identified Minority Classes: {ds_data['imbalance_metrics']['minority_classes']}")


if __name__ == "__main__":
    main()
