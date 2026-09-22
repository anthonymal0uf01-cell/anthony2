#!/usr/bin/env python3
"""
Build a controlled street-camera adaptation dataset from the public plate DB.

Only observations explicitly marked training_eligible are admitted.
The exporter balances cameras/classes, preserves provenance, and never promotes
machine OCR into verified registration truth.
"""
from __future__ import annotations
import argparse, json, shutil, sqlite3, hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEFAULT_DB=HERE/"au_public_plates.sqlite"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",type=Path,default=DEFAULT_DB)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--min-quality",type=float,default=.52)
    ap.add_argument("--max-per-camera",type=int,default=80)
    ap.add_argument("--max-per-class",type=int,default=500)
    ap.add_argument("--max-total",type=int,default=2500)
    a=ap.parse_args()

    con=sqlite3.connect(a.db)
    con.row_factory=sqlite3.Row
    rows=con.execute("""
      SELECT svo.*,m.local_path,m.license,m.source_page_url,m.image_url,
             cf.camera_id,cf.camera_title,cf.camera_region,cf.camera_view,cf.direction,cf.fetched_at
      FROM street_vehicle_observations svo
      JOIN media m ON m.media_id=svo.media_id
      JOIN camera_frames cf ON cf.frame_id=svo.frame_id
      WHERE svo.training_eligible=1 AND svo.quality_score>=?
      ORDER BY svo.quality_score DESC, svo.detector_confidence DESC
    """,(a.min_quality,)).fetchall()

    a.out.mkdir(parents=True,exist_ok=True)
    imgdir=a.out/"images"; imgdir.mkdir(exist_ok=True)
    selected=[]
    per_camera={}
    per_class={}
    seen_hash=set()

    for r in rows:
        if len(selected)>=a.max_total: break
        cam=r["camera_id"]; cls=r["detector_class"] or "vehicle"
        if per_camera.get(cam,0)>=a.max_per_camera: continue
        if per_class.get(cls,0)>=a.max_per_class: continue
        h=r["crop_sha256"]
        if not h or h in seen_hash: continue
        src=Path(r["local_path"] or "")
        if not src.exists(): continue

        ext=src.suffix.lower() if src.suffix else ".jpg"
        dst=imgdir/(h[:24]+ext)
        if not dst.exists():
            shutil.copy2(src,dst)

        rec={
          "image":str(dst.relative_to(a.out)),
          "sha256":h,
          "perceptual_hash":r["crop_perceptual_hash"],
          "detector_class":cls,
          "detector_confidence":r["detector_confidence"],
          "quality_score":r["quality_score"],
          "quality_flags":json.loads(r["quality_flags_json"] or "[]"),
          "crop_width":r["crop_width"],
          "crop_height":r["crop_height"],
          "bbox_area_fraction":r["bbox_area_fraction"],
          "edge_contact":bool(r["edge_contact"]),
          "camera_id":cam,
          "camera_title":r["camera_title"],
          "camera_region":r["camera_region"],
          "camera_view":r["camera_view"],
          "direction":r["direction"],
          "captured_at":r["fetched_at"],
          "license":r["license"],
          "source_page_url":r["source_page_url"],
          "provisional_plate":r["provisional_plate"],
          "provisional_plate_confidence":r["provisional_plate_confidence"],
          "plate_label_status":"machine_read" if r["provisional_plate"] else "none",
          "training_use":{
             "vehicle_domain_adaptation":True,
             "plate_text_ground_truth":False
          }
        }
        selected.append(rec)
        seen_hash.add(h)
        per_camera[cam]=per_camera.get(cam,0)+1
        per_class[cls]=per_class.get(cls,0)+1

    with (a.out/"manifest.jsonl").open("w",encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec,ensure_ascii=False)+"\n")

    summary={
      "selected":len(selected),
      "classes":per_class,
      "cameras":len(per_camera),
      "min_quality":a.min_quality,
      "rules":{
        "training_eligible_required":True,
        "machine_plate_text_is_not_ground_truth":True,
        "dedupe":"sha256 exact + collector perceptual dedupe",
        "camera_cap":a.max_per_camera,
        "class_cap":a.max_per_class
      }
    }
    (a.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary))

if __name__=="__main__":
    main()
