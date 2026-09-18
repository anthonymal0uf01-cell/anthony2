#!/usr/bin/env python3
"""
Bootstrap public Hugging Face datasets for Site Gate Tracker v7 Specialist.

This script deliberately keeps datasets separated by training role. It does NOT
mix foreign plate syntax into the final NSW/Australian recogniser. Detector and
degradation datasets teach visual invariances; OCR/domain adaptation remains a
separate stage, and the user's corrected gate data is the final specialist stage.

Usage:
  python bootstrap_hf_data.py --out data/hf
  python bootstrap_hf_data.py --out data/hf --only detector
  python bootstrap_hf_data.py --out data/hf --only temporal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

SOURCES = [
    {
        "id": "justjuu/license-plate-detection",
        "role": "detector",
        "license": "CC-BY-4.0",
        "purpose": "General plate localisation; COCO-style bounding boxes.",
        "allow_ocr_labels": False,
    },
    {
        "id": "evan6007/TLPD",
        "role": "detector",
        "license": "MIT",
        "purpose": "Plate geometry/polygon localisation; useful for oblique plates.",
        "allow_ocr_labels": False,
    },
    {
        "id": "thasan3003/NightLPBD_Nighttime_License_Plates_of_Bangladesh",
        "role": "adverse",
        "license": "CC-BY-4.0",
        "purpose": "Night, low-light, glare and motion-blur robustness. Use visual degradation/localisation labels; do not teach Bangla plate grammar to NSW OCR.",
        "allow_ocr_labels": False,
    },
    {
        "id": "kv1388/FANVID-Face_and_License_Plate_Recognition_in_Low-Resolution_Videos",
        "role": "temporal",
        "license": "CC-BY-4.0",
        "purpose": "Low-resolution video plate tracks; temporal recovery and multi-frame fusion.",
        "allow_ocr_labels": True,
    },
    {
        "id": "prithivMLmods/Number-Plate-Recognition",
        "role": "ocr_seed",
        "license": "Apache-2.0",
        "purpose": "Small OCR seed set with plate text. Use only as weak generic Latin-alphanumeric warm-up.",
        "allow_ocr_labels": True,
    },
]

ROLE_ORDER = {
    "detector": 10,
    "adverse": 20,
    "temporal": 30,
    "ocr_seed": 40,
    "gate": 50,
}

def safe_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hf")
    ap.add_argument(
        "--only",
        choices=["all", "detector", "adverse", "temporal", "ocr_seed"],
        default="all",
        help="Download only one curriculum role.",
    )
    ap.add_argument("--revision", default="main")
    args = ap.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []

    selected = [s for s in SOURCES if args.only == "all" or s["role"] == args.only]
    for src in selected:
        dest = root / safe_name(src["id"])
        print(f"\n==> {src['id']} [{src['role']}] -> {dest}")
        path = snapshot_download(
            repo_id=src["id"],
            repo_type="dataset",
            revision=args.revision,
            local_dir=str(dest),
        )
        item = dict(src)
        item["local_path"] = str(Path(path).resolve())
        item["curriculum_order"] = ROLE_ORDER[src["role"]]
        manifest.append(item)

    (root / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {root / 'sources.json'}")
    print("\nTraining rule: generic datasets teach perception first; corrected gate data is the final domain-adaptation stage.")

if __name__ == "__main__":
    main()
