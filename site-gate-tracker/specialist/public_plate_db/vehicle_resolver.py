#!/usr/bin/env python3
"""
Vehicle resolver bridge for the Australian public plate corpus.

Purpose:
    plate + jurisdiction
        -> authorised provider result (plate -> VIN)
        -> RAV public lookup by VIN
        -> canonical vehicle record
        -> registry-enriched training label

This module does NOT attempt to bypass provider access controls and does not
contain any owner/person fields. Provider responses are imported from a JSON
file or stdin after being obtained under the provider's authorised API terms.

RAV enrichment is one VIN at a time and rate-limited.
"""
from __future__ import annotations
import argparse, json, re, sqlite3, time, urllib.parse, urllib.request, uuid
from html import unescape
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEFAULT_DB=HERE/"au_public_plates.sqlite"
RAV_BASE="https://www.rover.infrastructure.gov.au/RAVPublicSearch/"

RAV_FIELDS=[
 ("VIN","vin"),
 ("RAV Date of Entry","rav_date_of_entry"),
 ("Entry Pathway Sub-Category","entry_pathway_subcategory"),
 ("Approval Number","approval_number"),
 ("Approval Holder","approval_holder"),
 ("VCC","vehicle_category_code"),
 ("Vehicle Make","vehicle_make"),
 ("Vehicle Model","vehicle_model"),
 ("Authorised By Name","authorised_by_name"),
 ("Build Date","build_date"),
 ("GVM/ATM (kg)","gvm_atm_kg"),
 ("GTM","gtm_kg"),
 ("Tare","tare_kg"),
 ("Motive Power","motive_power"),
 ("Power","power_kw"),
 ("GCM","gcm_kg"),
 ("Seats","seating_capacity"),
 ("NVES Vehicle Type","nves_vehicle_type"),
 ("Carbon Dioxide Emissions (g/km)","co2_g_km"),
 ("Mass In Running Order (kg)","mass_in_running_order_kg"),
]
FIELD_LABELS=[x[0] for x in RAV_FIELDS]

def norm_plate(s:str)->str:
    return re.sub(r"[^A-Z0-9]","",(s or "").upper())[:10]

def norm_vin(s:str)->str:
    v=re.sub(r"[^A-Z0-9]","",(s or "").upper())
    if len(v)!=17 or any(c in v for c in "IOQ"):
        raise ValueError("VIN must be 17 characters and cannot contain I, O or Q")
    return v

def visible_text(html:str)->str:
    html=re.sub(r"(?is)<script.*?</script>|<style.*?</style>"," ",html)
    html=re.sub(r"(?s)<[^>]+>","\n",html)
    lines=[re.sub(r"\s+"," ",unescape(x)).strip() for x in html.splitlines()]
    return "\n".join(x for x in lines if x)

def parse_rav(text:str)->dict:
    # Public page renders labels followed by values. We search between known labels.
    out={}
    positions=[]
    for label,key in RAV_FIELDS:
        for m in re.finditer(re.escape(label)+r"\s*",text,re.I):
            positions.append((m.start(),m.end(),label,key))
            break
    positions.sort()
    for i,(_,end,label,key) in enumerate(positions):
        nxt=positions[i+1][0] if i+1<len(positions) else len(text)
        val=text[end:nxt].strip().split("\n")[0].strip(" :-")
        # Avoid swallowing public-search boilerplate if a field is absent.
        if len(val)>200 or val.lower().startswith("the register of approved vehicles"):
            val=""
        if val:
            out[key]=val
    # VCC values often include code + description and are useful as-is.
    return out

def fetch_rav(vin:str, delay:float=1.0)->dict:
    vin=norm_vin(vin)
    time.sleep(max(0.0,delay))
    url=RAV_BASE+"?"+urllib.parse.urlencode({"vinno":vin})
    req=urllib.request.Request(url,headers={"User-Agent":"AU-Vehicle-Knowledge/1.0"})
    with urllib.request.urlopen(req,timeout=30) as r:
        body=r.read().decode("utf-8","replace")
    txt=visible_text(body)
    if "does not contain a record for this search" in txt.lower():
        return {"vin":vin,"found":False,"source_url":url}
    fields=parse_rav(txt)
    if not fields.get("vin"):
        fields["vin"]=vin
    fields["found"]=True
    fields["source_url"]=url
    fields["fetched_at"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    return fields

def connect(db:Path):
    con=sqlite3.connect(db)
    con.row_factory=sqlite3.Row
    return con

def import_provider(con, record:dict):
    """
    Expected provider-neutral JSON:
      {
        "provider": "InfoAgent",
        "plate": "ABC12D",
        "jurisdiction": "NSW",
        "vin": "17CHARVIN...",
        "verified": true,
        "confidence": 1.0,
        "provider_reference": "...",
        "vehicle": {
          "make": "...", "model": "...", "variant": "...", "series": "...",
          "body_type": "...", "vehicle_class": "...", "colour": "...",
          "build_year": 2025, "gvm_kg": 0, "gcm_kg": 0, "tare_kg": 0
        }
      }
    """
    plate=norm_plate(record["plate"])
    state=str(record["jurisdiction"]).upper().strip()
    vin=norm_vin(record["vin"]) if record.get("vin") else None
    verified=bool(record.get("verified",False))
    rid="res:"+uuid.uuid4().hex
    now=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    vehicle_id=("vin:"+vin) if vin else None
    v=record.get("vehicle") or {}
    if vehicle_id:
        con.execute("""INSERT INTO vehicles
        (vehicle_id,vin,make,model,variant,series,build_year,body_type,vehicle_class,
         colour,gvm_kg,gcm_kg,tare_kg,source_of_truth,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(vehicle_id) DO UPDATE SET
          make=COALESCE(excluded.make,vehicles.make),
          model=COALESCE(excluded.model,vehicles.model),
          variant=COALESCE(excluded.variant,vehicles.variant),
          series=COALESCE(excluded.series,vehicles.series),
          build_year=COALESCE(excluded.build_year,vehicles.build_year),
          body_type=COALESCE(excluded.body_type,vehicles.body_type),
          vehicle_class=COALESCE(excluded.vehicle_class,vehicles.vehicle_class),
          colour=COALESCE(excluded.colour,vehicles.colour),
          gvm_kg=COALESCE(excluded.gvm_kg,vehicles.gvm_kg),
          gcm_kg=COALESCE(excluded.gcm_kg,vehicles.gcm_kg),
          tare_kg=COALESCE(excluded.tare_kg,vehicles.tare_kg),
          source_of_truth=excluded.source_of_truth,
          updated_at=excluded.updated_at
        """,(vehicle_id,vin,v.get("make"),v.get("model"),v.get("variant"),v.get("series"),
        v.get("build_year"),v.get("body_type"),v.get("vehicle_class"),v.get("colour"),
        v.get("gvm_kg"),v.get("gcm_kg"),v.get("tare_kg"),
        record.get("provider","provider"),now))
        con.execute("""INSERT OR REPLACE INTO vehicle_plates
        (vehicle_id,plate_text,jurisdiction,evidence_source)
        VALUES(?,?,?,?)""",(vehicle_id,plate,state,record.get("provider","provider")))
    con.execute("""INSERT INTO vehicle_resolutions
      (resolution_id,plate_text,jurisdiction,vin,vehicle_id,resolution_status,
       provider,provider_reference,confidence,resolved_at,raw_response_json)
      VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
      (rid,plate,state,vin,vehicle_id,
       "provider_verified" if verified else "provider_candidate",
       record.get("provider"),record.get("provider_reference"),
       record.get("confidence"),now,json.dumps(record,ensure_ascii=False)))
    con.commit()
    return vehicle_id,vin

def upsert_rav(con, r:dict):
    if not r.get("found"): return
    num=lambda k: float(re.sub(r"[^0-9.]","",str(r.get(k,"")))) if re.search(r"[0-9]",str(r.get(k,""))) else None
    integer=lambda k: int(num(k)) if num(k) is not None else None
    vin=norm_vin(r["vin"])
    con.execute("""INSERT INTO rav_records
    (vin,rav_date_of_entry,entry_pathway_subcategory,approval_number,approval_holder,
     vehicle_category_code,vehicle_make,vehicle_model,authorised_by_name,build_date,
     gvm_atm_kg,gtm_kg,tare_kg,motive_power,power_kw,gcm_kg,seating_capacity,
     nves_vehicle_type,co2_g_km,mass_in_running_order_kg,source_url,fetched_at,raw_fields_json)
    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(vin) DO UPDATE SET
      rav_date_of_entry=excluded.rav_date_of_entry,
      entry_pathway_subcategory=excluded.entry_pathway_subcategory,
      approval_number=excluded.approval_number,
      approval_holder=excluded.approval_holder,
      vehicle_category_code=excluded.vehicle_category_code,
      vehicle_make=excluded.vehicle_make,
      vehicle_model=excluded.vehicle_model,
      authorised_by_name=excluded.authorised_by_name,
      build_date=excluded.build_date,
      gvm_atm_kg=excluded.gvm_atm_kg,gtm_kg=excluded.gtm_kg,tare_kg=excluded.tare_kg,
      motive_power=excluded.motive_power,power_kw=excluded.power_kw,gcm_kg=excluded.gcm_kg,
      seating_capacity=excluded.seating_capacity,nves_vehicle_type=excluded.nves_vehicle_type,
      co2_g_km=excluded.co2_g_km,mass_in_running_order_kg=excluded.mass_in_running_order_kg,
      source_url=excluded.source_url,fetched_at=excluded.fetched_at,raw_fields_json=excluded.raw_fields_json
    """,(vin,r.get("rav_date_of_entry"),r.get("entry_pathway_subcategory"),r.get("approval_number"),
    r.get("approval_holder"),r.get("vehicle_category_code"),r.get("vehicle_make"),r.get("vehicle_model"),
    r.get("authorised_by_name"),r.get("build_date"),num("gvm_atm_kg"),num("gtm_kg"),num("tare_kg"),
    r.get("motive_power"),num("power_kw"),num("gcm_kg"),integer("seating_capacity"),
    r.get("nves_vehicle_type"),num("co2_g_km"),num("mass_in_running_order_kg"),
    r.get("source_url"),r.get("fetched_at"),json.dumps(r,ensure_ascii=False)))
    # Merge public RAV facts into canonical vehicle record without overwriting richer provider fields.
    vid="vin:"+vin
    con.execute("""INSERT INTO vehicles
    (vehicle_id,vin,make,model,vehicle_class,gvm_kg,gcm_kg,tare_kg,source_of_truth,updated_at)
    VALUES(?,?,?,?,?,?,?,?,?,?)
    ON CONFLICT(vehicle_id) DO UPDATE SET
      make=COALESCE(vehicles.make,excluded.make),
      model=COALESCE(vehicles.model,excluded.model),
      vehicle_class=COALESCE(vehicles.vehicle_class,excluded.vehicle_class),
      gvm_kg=COALESCE(vehicles.gvm_kg,excluded.gvm_kg),
      gcm_kg=COALESCE(vehicles.gcm_kg,excluded.gcm_kg),
      tare_kg=COALESCE(vehicles.tare_kg,excluded.tare_kg),
      updated_at=excluded.updated_at
    """,(vid,vin,r.get("vehicle_make"),r.get("vehicle_model"),r.get("vehicle_category_code"),
    num("gvm_atm_kg"),num("gcm_kg"),num("tare_kg"),"RAV",
    time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())))
    con.commit()

def make_training_labels(con):
    rows=con.execute("""
      SELECT po.observation_id,po.plate_text,po.jurisdiction,vr.vin,vr.vehicle_id,
             v.make,v.model,v.variant,v.series,v.body_type,v.vehicle_class,
             v.gvm_kg,v.gcm_kg,v.tare_kg,rr.vehicle_make,rr.vehicle_model,rr.vehicle_category_code
      FROM plate_observations po
      JOIN vehicle_resolutions vr
        ON vr.plate_text=po.plate_text AND vr.jurisdiction=po.jurisdiction
       AND vr.resolution_status IN ('provider_verified','manual_verified')
      LEFT JOIN vehicles v ON v.vehicle_id=vr.vehicle_id
      LEFT JOIN rav_records rr ON rr.vin=vr.vin
      WHERE po.plate_text IS NOT NULL AND vr.vin IS NOT NULL
    """).fetchall()
    now=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); n=0
    for r in rows:
        tid="train:"+r["observation_id"]+":"+r["vin"]
        prov={"plate":"registry_verified","vin":"provider_verified","rav":bool(r["vehicle_make"])}
        con.execute("""INSERT OR REPLACE INTO training_labels
        (training_label_id,observation_id,vehicle_id,vin,plate_text,jurisdiction,
         make,model,variant,series,body_type,vehicle_class,gvm_kg,gcm_kg,tare_kg,
         label_quality,provenance_json,created_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (tid,r["observation_id"],r["vehicle_id"],r["vin"],r["plate_text"],r["jurisdiction"],
         r["make"] or r["vehicle_make"],r["model"] or r["vehicle_model"],r["variant"],r["series"],
         r["body_type"],r["vehicle_class"] or r["vehicle_category_code"],r["gvm_kg"],r["gcm_kg"],r["tare_kg"],
         "registry_enriched" if r["vehicle_make"] else "vin_verified",
         json.dumps(prov),now))
        n+=1
    con.commit(); return n

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",type=Path,default=DEFAULT_DB)
    sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("import-provider");p.add_argument("json_file",type=Path)
    p=sub.add_parser("rav");p.add_argument("vin");p.add_argument("--delay",type=float,default=1.0)
    sub.add_parser("build-training-labels")
    a=ap.parse_args();con=connect(a.db)
    if a.cmd=="import-provider":
        rec=json.loads(a.json_file.read_text(encoding="utf-8"))
        vid,vin=import_provider(con,rec);print(json.dumps({"vehicle_id":vid,"vin":vin}))
    elif a.cmd=="rav":
        r=fetch_rav(a.vin,a.delay);upsert_rav(con,r);print(json.dumps(r,indent=2))
    elif a.cmd=="build-training-labels":
        print("training labels",make_training_labels(con))

if __name__=="__main__": main()
