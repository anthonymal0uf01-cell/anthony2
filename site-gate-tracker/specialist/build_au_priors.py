#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,zipfile
from collections import Counter
from pathlib import Path

def num(v):
    try:return float(str(v).replace(",","").strip())
    except:return 0.0

def csv_rows_bytes(data:bytes):
    text=data.decode("utf-8-sig",errors="replace")
    return list(csv.DictReader(text.splitlines()))

def aggregate(rows,key_hints,count_hints):
    if not rows:return {}
    fields=list(rows[0])
    key=next((f for h in key_hints for f in fields if h in f.lower()),None)
    count=next((f for h in count_hints for f in fields if h in f.lower()),None)
    if not key:return {}
    c=Counter()
    for r in rows:c[str(r.get(key,"")).strip()]+=num(r.get(count,1)) if count else 1
    total=sum(c.values()) or 1
    return {"key":key,"count":count,"total":total,"prior":{k:v/total for k,v in c.most_common() if k}}

def read_zip_csvs(path:Path):
    out=[]
    if not path.exists():return out
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.lower().endswith(".csv"):
                try:out.extend(csv_rows_bytes(z.read(n)))
                except Exception:pass
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",type=Path,help="data/australia from bootstrap_au_data.py")
    ap.add_argument("--out",type=Path,default=Path("data/au_priors.json"))
    args=ap.parse_args();official=args.root/"official"
    result={}
    rva=official/"road_vehicles_australia_2025.csv"
    if rva.exists():
        rows=list(csv.DictReader(rva.open(encoding="utf-8-sig",errors="replace")))
        result["fleet_by_type"]=aggregate(rows,["vehicle_type","vehicle type"],["no_vehicles","count","vehicles","total"])
        nsw=[r for r in rows if str(r.get("state_abb","")).upper()=="NSW"]
        result["nsw_fleet_by_type"]=aggregate(nsw,["vehicle_type","vehicle type"],["no_vehicles","count","vehicles","total"])
    nsw_snapshot=read_zip_csvs(official/"tfnsw_vehicle_registrations_snapshot_2026.zip")
    result["nsw_registration_vehicle_type"]=aggregate(
        nsw_snapshot,["vehicle type","vehicle_type","body type","body_type","vehicle class"],
        ["count","vehicles","total","number","registrations"])
    result["nsw_registration_make"]=aggregate(
        nsw_snapshot,["make","manufacturer"],
        ["count","vehicles","total","number","registrations"])
    usage=read_zip_csvs(official/"tfnsw_registered_vehicles_by_usage_2026.zip")
    result["nsw_vehicle_usage"]=aggregate(
        usage,["usage","vehicle usage","registration usage","use"],
        ["count","vehicles","total","number","registrations"])
    heavy=read_zip_csvs(official/"tfnsw_heavy_vehicle_configuration_2026.zip")
    result["nsw_heavy_configuration"]=aggregate(heavy,["configuration","vehicle type","body type","description"],["count","vehicles","total","number"])
    vic=[]
    vdir=official/"victoria"
    if vdir.exists():
        for p in vdir.iterdir():
            if p.suffix.lower()==".zip": vic.extend(read_zip_csvs(p))
            elif p.suffix.lower()==".csv":
                try:vic.extend(list(csv.DictReader(p.open(encoding="utf-8-sig",errors="replace"))))
                except Exception:pass
    result["vic_austroads_class"]=aggregate(vic,["vehicle_class","vehicle class"],["volume","count","total"])
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({k:{"categories":len(v.get("prior",{})),"total":v.get("total",0)} for k,v in result.items()},indent=2))
if __name__=="__main__":main()
