#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

TFNSW_CAMERAS = "https://opendata.transport.nsw.gov.au/data/dataset/b0212311-b0da-4363-8dc3-825fe10941b2/resource/cc776d1a-d96c-4ae4-a465-c380a53717c9/download/livetrafficcamera.json"
NSW_HEAVY_2026 = "https://opendata.transport.nsw.gov.au/data/dataset/2342b8e4-d4fc-4549-82de-b36380ea46f2/resource/15ae0f12-ba3f-4189-9f67-87421141b47b/download/tfnsw_registered_heavy_vehicles_by_configuration_snapshot_2026.zip"
RVA_2025 = "https://data.gov.au/data/dataset/f6e0a290-7d47-4b88-ac3b-34824b0ab334/resource/87bd686c-29cf-474e-a14f-4ec4f576dbff/download/rva-2025-mvs-vehtype-streg-mtvpwr-rpc.csv"
VIC_PACKAGE_API = "https://discover.data.vic.gov.au/api/3/action/package_show?id=a7c112d6-668f-4777-a61b-80f646f3f2b9"

def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print("exists", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("download", url, "->", dest)
    req = urllib.request.Request(url, headers={"User-Agent":"SiteGateTracker/7 AustraliaSpecialist"})
    with urllib.request.urlopen(req, timeout=90) as r, dest.open("wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

def fetch_json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent":"SiteGateTracker/7 AustraliaSpecialist"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)

def get_vic_2026_resources() -> list[dict]:
    try:
        data = fetch_json(VIC_PACKAGE_API)
        resources = data.get("result",{}).get("resources",[])
        out=[]
        for r in resources:
            name=(r.get("name") or "").lower()
            fmt=(r.get("format") or "").upper()
            if "2026" in name and fmt in {"ZIP","CSV"} and r.get("url"):
                out.append({"name":r.get("name"),"format":fmt,"url":r["url"]})
        return out
    except Exception as e:
        print("warning: could not enumerate DataVic resources:", e)
        return []

def get_roboflow(out: Path) -> None:
    key=os.getenv("ROBOFLOW_API_KEY")
    if not key:
        print("ROBOFLOW_API_KEY not set; skipping Roboflow datasets.")
        return
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow first: pip install roboflow")
        return
    rf=Roboflow(api_key=key)
    jobs=[
        ("alpr-dld41","au-nsw-white",2,"nsw_white"),
        ("gattondieselmaker","australian-license-plates-j0eml",2,"au_multistate"),
    ]
    for workspace, project, version, folder in jobs:
        dest=out/"roboflow"/folder
        if dest.exists() and any(dest.iterdir()):
            print("exists",dest);continue
        print("roboflow",workspace,project,"v",version)
        rf.workspace(workspace).project(project).version(version).download("yolov8", location=str(dest))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,default=Path("data/australia"))
    ap.add_argument("--roboflow",action="store_true")
    args=ap.parse_args()
    out=args.out
    out.mkdir(parents=True,exist_ok=True)

    download(TFNSW_CAMERAS,out/"official"/"nsw_live_traffic_cameras.json")
    download(NSW_HEAVY_2026,out/"official"/"tfnsw_heavy_vehicle_configuration_2026.zip")
    download(RVA_2025,out/"official"/"road_vehicles_australia_2025.csv")

    vic=[]
    for r in get_vic_2026_resources():
        safe="".join(c if c.isalnum() or c in "-_." else "_" for c in r["name"])
        dest=out/"official"/"victoria"/safe
        try:
            download(r["url"],dest)
            vic.append({**r,"local_path":str(dest)})
        except Exception as e:
            print("warning: DataVic download failed:",r["name"],e)

    if args.roboflow:
        get_roboflow(out)

    manifest={
        "created_by":"Site Gate Tracker v7 Australia Specialist",
        "official_sources":{
            "tfnsw_live_cameras":{"path":str(out/"official"/"nsw_live_traffic_cameras.json"),"use":"vehicle-domain metadata; mask plates before retaining imagery"},
            "nsw_heavy_2026":{"path":str(out/"official"/"tfnsw_heavy_vehicle_configuration_2026.zip"),"use":"heavy configuration priors"},
            "road_vehicles_au_2025":{"path":str(out/"official"/"road_vehicles_australia_2025.csv"),"use":"fleet priors"},
            "victoria_telemetry_2026":vic,
        },
        "roboflow_requested":bool(args.roboflow),
    }
    (out/"bootstrap_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))

if __name__=="__main__":
    main()
