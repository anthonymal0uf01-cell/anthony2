#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path

def sha256(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

def load(p):return json.loads(p.read_text(encoding="utf-8"))

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=Path("site-gate-tracker/specialist"));ap.add_argument("--manifest",type=Path,default=None);a=ap.parse_args()
    w=a.root/"weights"
    required=["au_ocr_seed.pt","au_ocr_seed.onnx","au_ocr_seed.metrics.json",
              "au_plate_detector.pt","au_plate_detector.onnx","au_plate_detector.metrics.json",
              "au_vehicle_profile.json","au_official_priors.json"]
    missing=[x for x in required if not (w/x).exists()]
    if missing:raise SystemExit("PHASE2 INCOMPLETE missing: "+", ".join(missing))
    o=load(w/"au_ocr_seed.metrics.json");d=load(w/"au_plate_detector.metrics.json");v=load(w/"au_vehicle_profile.json");p=load(w/"au_official_priors.json")
    failures=[]
    if o.get("exact_match",0)<.95:failures.append(f"OCR exact {o.get('exact_match',0):.3f}<.95")
    if o.get("worst_hard_slice",0)<.90:failures.append(f"OCR hard slice {o.get('worst_hard_slice',0):.3f}<.90")
    cam=(o.get("camera_domain") or {}).get("exact_match",0)
    if cam<.80:failures.append(f"OCR camera-domain {cam:.3f}<.80")
    if o.get("brier_calibrated",1)>o.get("brier_raw",0)+.002:failures.append("OCR calibration materially worsened Brier score")
    if not o.get("calibration_selection"):failures.append("OCR missing validation-safe calibration selection record")
    if d.get("map50",0)<.60:failures.append(f"detector mAP50 {d.get('map50',0):.3f}<.60")
    if d.get("recall",0)<.60:failures.append(f"detector recall {d.get('recall',0):.3f}<.60")
    if d.get("evaluation_split")!="test":failures.append("detector was not promoted from untouched test split")
    if v.get("frames",0)<8 or v.get("detections",0)<1:failures.append("Australian vehicle profile insufficient")
    catalog=load(a.root/"AU_DATA_CATALOG_2026.json")
    prod={x["id"]:x for x in catalog.get("sources",[]) if not str(x.get("policy","")).startswith("research")}
    for src in ["justjuu_plate_detection","tfnsw_live_cameras"]:
        if src not in prod:failures.append("source policy missing production source "+src)
    if failures:raise SystemExit("PHASE2 PROMOTION FAILED\n- "+"\n- ".join(failures))
    manifest={
      "phase":"2","status":"PROMOTED","objective":"Australian specialist perception bundle",
      "metrics":{"ocr":o,"detector":d,
                 "vehicle_profile":{"frames":v.get("frames"),"detections":v.get("detections"),"visual_prior":v.get("visual_prior")},
                 "official_prior_sections":sorted(p.keys())},
      "artifacts":{x:{"bytes":(w/x).stat().st_size,"sha256":sha256(w/x)} for x in required},
      "production_sources":[prod[x] for x in ["justjuu_plate_detection","tfnsw_live_cameras"]],
      "optional_enrichment_sources":[x for x in catalog.get("sources",[]) if x.get("id")=="tlpd"],
      "research_only_excluded":[x["id"] for x in catalog.get("sources",[]) if str(x.get("policy","")).startswith("research")],
    }
    out=a.manifest or a.root/"PHASE2_BUNDLE_2026.json";out.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"status":"PROMOTED","manifest":str(out),"ocr_exact":o["exact_match"],"detector_map50":d["map50"],"detector_recall":d["recall"],"vehicle_frames":v["frames"]},indent=2))
if __name__=="__main__":main()
