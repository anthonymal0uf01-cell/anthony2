#!/usr/bin/env python3
"""
Build a licence-auditable Australian public plate corpus.

This tool:
1) creates the SQLite schema;
2) imports the curated source catalogue;
3) can harvest Wikimedia Commons categories with per-file licence metadata;
4) can import local Roboflow YOLO exports after you download them lawfully.

It deliberately does NOT store owner/person information.
It never promotes OCR guesses to verified labels automatically.
"""
from __future__ import annotations
import argparse, hashlib, html, json, re, sqlite3, time, urllib.parse, urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCHEMA=HERE/"schema.sql"
SOURCES=HERE/"sources.json"
COMMONS_API="https://commons.wikimedia.org/w/api.php"
PLATE_PATTERNS=[
    re.compile(r"(?<![A-Z0-9])([A-Z]{3}[ -]?[0-9]{3})(?![A-Z0-9])",re.I),
    re.compile(r"(?<![A-Z0-9])([A-Z0-9]{1,4}[ .·-][A-Z0-9]{2,4})(?![A-Z0-9])",re.I),
    re.compile(r"(?<![A-Z0-9])([A-Z]{2}[0-9]{2}[A-Z]{2})(?![A-Z0-9])",re.I),
]
JURIS={
 "New South Wales":"NSW","Queensland":"QLD","Victoria":"VIC",
 "South Australia":"SA","Tasmania":"TAS","Northern Territory":"NT",
 "Australian Capital Territory":"ACT","Western Australia":"WA"
}

def req(params):
    url=COMMONS_API+"?"+urllib.parse.urlencode(params)
    r=urllib.request.Request(url,headers={"User-Agent":"AUPlateCorpus/1.0 research-contact"})
    with urllib.request.urlopen(r,timeout=30) as f:
        return json.load(f)

def clean(v):
    if isinstance(v,dict): v=v.get("value","")
    return html.unescape(re.sub(r"<[^>]+>"," ",str(v or ""))).strip()

def candidate_plate(text):
    s=clean(text).upper()
    for rx in PLATE_PATTERNS:
        m=rx.search(s)
        if m:
            return re.sub(r"[^A-Z0-9]","",m.group(1))
    return ""

def jurisdiction_from(title,desc=""):
    s=(title+" "+desc).lower()
    for k,v in JURIS.items():
        if k.lower() in s: return v
    return ""

def init_db(path):
    con=sqlite3.connect(path)
    con.executescript(SCHEMA.read_text(encoding="utf-8"))
    catalog=json.loads(SOURCES.read_text(encoding="utf-8"))
    now=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    for s in catalog["sources"]:
        con.execute("""INSERT OR REPLACE INTO sources
        (source_id,name,kind,source_url,license,production_policy,retrieved_at)
        VALUES(?,?,?,?,?,?,?)""",(s["id"],s["name"],s["kind"],s["url"],
        str(s.get("license","")),s.get("production_policy",""),now))
    con.commit()
    return con

def category_files(category):
    cont={}
    while True:
        p={"action":"query","format":"json","generator":"categorymembers",
           "gcmtitle":"Category:"+category,"gcmtype":"file","gcmlimit":"100",
           "prop":"imageinfo","iiprop":"url|size|extmetadata"}
        p.update(cont)
        data=req(p)
        for page in data.get("query",{}).get("pages",{}).values():
            yield page
        if "continue" not in data: break
        cont=data["continue"]

def import_commons(con,category,source_id):
    now=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    n=0
    for page in category_files(category):
        ii=(page.get("imageinfo") or [{}])[0]; meta=ii.get("extmetadata",{})
        title=page.get("title",""); desc=clean(meta.get("ImageDescription"))
        lic=clean(meta.get("LicenseShortName")); licurl=clean(meta.get("LicenseUrl"))
        url=ii.get("url",""); pageurl=ii.get("descriptionurl","")
        media_id="commons:"+str(page.get("pageid") or hashlib.sha1(title.encode()).hexdigest())
        jur=jurisdiction_from(title,desc)
        con.execute("""INSERT OR REPLACE INTO media
        (media_id,source_id,source_page_url,image_url,jurisdiction,author,license,
         license_url,width,height,captured_at,imported_at,metadata_json)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (media_id,source_id,pageurl,url,jur,clean(meta.get("Artist")),lic,licurl,
         ii.get("width"),ii.get("height"),clean(meta.get("DateTimeOriginal")),now,
         json.dumps({"title":title,"description":desc},ensure_ascii=False)))
        plate=candidate_plate(title+" "+desc)
        oid=media_id+":plate0"
        con.execute("""INSERT OR REPLACE INTO plate_observations
        (observation_id,media_id,plate_text,text_status,jurisdiction,notes)
        VALUES(?,?,?,?,?,?)""",
        (oid,media_id,plate or None,"title_inferred" if plate else "none",jur,
         "Title/description inference only; human verification required before OCR training."))
        n+=1
        if n%100==0: con.commit()
    con.commit()
    return n

def import_yolo(con,root,source_id,jurisdiction=""):
    root=Path(root); names={}
    yml=next(iter(root.rglob("data.yaml")),None)
    if yml:
        txt=yml.read_text(encoding="utf-8",errors="ignore")
        for m in re.finditer(r"^\s*(\d+)\s*:\s*['\"]?([^'\"\n]+)",txt,re.M):
            names[int(m.group(1))]=m.group(2).strip()
    now=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()); n=0
    for lab in root.rglob("*.txt"):
        if "labels" not in lab.parts: continue
        img=None
        for ext in [".jpg",".jpeg",".png",".webp"]:
            cand=Path(str(lab).replace("/labels/","/images/")).with_suffix(ext)
            if cand.exists(): img=cand; break
        if not img: continue
        media_id=source_id+":"+hashlib.sha1(str(img).encode()).hexdigest()
        con.execute("""INSERT OR REPLACE INTO media
        (media_id,source_id,local_path,jurisdiction,imported_at,metadata_json)
        VALUES(?,?,?,?,?,?)""",(media_id,source_id,str(img),jurisdiction,now,
        json.dumps({"label_file":str(lab)})))
        for i,line in enumerate(lab.read_text().splitlines()):
            p=line.split()
            if len(p)<5: continue
            cls=int(float(p[0])); cx,cy,w,h=map(float,p[1:5])
            con.execute("""INSERT OR REPLACE INTO plate_observations
            (observation_id,media_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2,
             plate_text,text_status,jurisdiction,plate_style,notes)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (f"{media_id}:plate{i}",media_id,cx-w/2,cy-h/2,cx+w/2,cy+h/2,
             None,"none",jurisdiction,names.get(cls,str(cls)),
             "Detection label only; OCR text not supplied by source."))
        n+=1
    con.commit(); return n

def stats(con):
    for q in [
      "select count(*) from media",
      "select count(*) from plate_observations",
      "select count(*) from plate_observations where plate_text is not null",
      "select count(*) from plate_observations where text_status in ('human_verified','registry_verified','synthetic_exact')"
    ]:
        print(con.execute(q).fetchone()[0])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",type=Path,default=HERE/"au_public_plates.sqlite")
    ap.add_argument("--commons-category")
    ap.add_argument("--source-id",default="commons_au_automobiles")
    ap.add_argument("--roboflow-root",type=Path)
    ap.add_argument("--jurisdiction",default="")
    ap.add_argument("--stats",action="store_true")
    a=ap.parse_args()
    con=init_db(a.db)
    if a.commons_category:
        print("commons imported",import_commons(con,a.commons_category,a.source_id))
    if a.roboflow_root:
        print("yolo media imported",import_yolo(con,a.roboflow_root,a.source_id,a.jurisdiction))
    if a.stats: stats(con)
    print(a.db)

if __name__=="__main__": main()
