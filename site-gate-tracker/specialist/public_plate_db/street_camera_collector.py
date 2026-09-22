#!/usr/bin/env python3
"""
TfNSW live street-camera -> Australian vehicle corpus collector.

Pipeline:
  current camera index
    -> repeated JPEG snapshots
    -> unchanged-frame rejection
    -> vehicle detector
    -> per-vehicle crop
    -> Australian plate detector
    -> Australian OCR
    -> SQLite observation

Machine OCR is stored as machine_read only. It never becomes registry truth
until a separate authorised plate->VIN provider resolves it.

The collector is bounded by design:
- configurable camera count and sample interval
- unchanged frames skipped
- near-duplicate vehicle crops skipped
- only vehicle crops are retained by default

Environment:
  TFNSW_API_KEY  optional, for https://api.transport.nsw.gov.au/v1/live/cameras
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, re, sqlite3, sys, time, uuid
from pathlib import Path
from typing import Any
import urllib.parse, urllib.request, urllib.error

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
SPECIALIST=HERE.parent
DEFAULT_DB=HERE/"au_public_plates.sqlite"
DEFAULT_OUT=HERE/"street_crops"
PUBLIC_INDEX="https://data.livetraffic.com/cameras/traffic-cam.json"
API_INDEX="https://api.transport.nsw.gov.au/v1/live/cameras"
PLATE_MODEL=SPECIALIST/"weights"/"au_plate_detector.pt"
OCR_MODEL=SPECIALIST/"weights"/"au_ocr_seed.onnx"
ALPHABET="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
VEHICLE_CLASSES={"car","truck","bus","motorcycle"}

def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())

def request(url:str, *, api_key:str|None=None, timeout=25)->bytes:
    h={
      "User-Agent":"AU-Vehicle-Knowledge/1.0",
      "Accept":"application/json,image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
      "Referer":"https://www.livetraffic.com/",
      "Cache-Control":"no-cache",
    }
    if api_key:
        h["Authorization"]="apikey "+api_key
    req=urllib.request.Request(url,headers=h)
    with urllib.request.urlopen(req,timeout=timeout) as r:
        return r.read()

def load_index():
    key=os.getenv("TFNSW_API_KEY","").strip()
    errors=[]
    if key:
        try:
            return json.loads(request(API_INDEX,api_key=key))
        except Exception as e:
            errors.append("api:"+repr(e))
    try:
        return json.loads(request(PUBLIC_INDEX))
    except Exception as e:
        errors.append("public:"+repr(e))
    raise RuntimeError("Could not load TfNSW camera index: "+"; ".join(errors))

def cameras(index:dict[str,Any], regions:set[str]|None=None):
    out=[]
    for f in index.get("features",[]):
        p=f.get("properties") or {}; g=f.get("geometry") or {}
        coords=g.get("coordinates") or [None,None]
        if not p.get("href"): continue
        if regions and str(p.get("region","")) not in regions: continue
        out.append({
          "id":str(f.get("id") or hashlib.sha1(str(p).encode()).hexdigest()),
          "title":str(p.get("title","")),
          "view":str(p.get("view","")),
          "region":str(p.get("region","")),
          "direction":str(p.get("direction","")),
          "href":str(p.get("href")),
          "longitude":coords[0] if len(coords)>0 else None,
          "latitude":coords[1] if len(coords)>1 else None,
        })
    return out

def sha(b:bytes): return hashlib.sha256(b).hexdigest()

def ahash(im:Image.Image)->str:
    a=np.asarray(im.convert("L").resize((8,8),Image.Resampling.BILINEAR),dtype=np.float32)
    bits=(a>a.mean()).reshape(-1)
    return "".join(f"{int(''.join('1' if x else '0' for x in bits[i:i+4]),2):x}" for i in range(0,64,4))

def hamming(a:str,b:str)->int:
    try:
        return sum((int(x,16)^int(y,16)).bit_count() for x,y in zip(a,b))
    except Exception: return 64

def init_db(path:Path):
    con=sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript((HERE/"schema.sql").read_text(encoding="utf-8"))
    con.execute("""INSERT OR IGNORE INTO sources
      (source_id,name,kind,source_url,license,production_policy,retrieved_at)
      VALUES(?,?,?,?,?,?,?)""",(
        "tfnsw_live_cameras","Live Traffic NSW cameras","current_vehicle_appearance",
        PUBLIC_INDEX,"Creative Commons Attribution",
        "Current vehicle-domain observations. OCR remains machine_read until independently verified.",utc()
      ))
    con.commit()
    return con

class Models:
    def __init__(self, vehicle_model:str="yolo26n.pt"):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Install ultralytics: pip install ultralytics onnxruntime pillow numpy") from e
        self.vehicle=YOLO(vehicle_model)
        if not PLATE_MODEL.exists():
            raise FileNotFoundError(f"missing {PLATE_MODEL}")
        self.plate=YOLO(str(PLATE_MODEL))
        try:
            import onnxruntime as ort
        except Exception as e:
            raise RuntimeError("Install onnxruntime") from e
        if not OCR_MODEL.exists():
            raise FileNotFoundError(f"missing {OCR_MODEL}")
        providers=["CUDAExecutionProvider","CPUExecutionProvider"]
        avail=set(ort.get_available_providers())
        providers=[p for p in providers if p in avail]
        self.ocr=ort.InferenceSession(str(OCR_MODEL),providers=providers or None)
        self.ocr_in=self.ocr.get_inputs()[0].name
        self.ocr_out=self.ocr.get_outputs()[0].name

    @staticmethod
    def ocr_input(im:Image.Image):
        im=im.convert("L")
        im.thumbnail((160,48),Image.Resampling.LANCZOS)
        canvas=Image.new("L",(160,48),127)
        canvas.paste(im,((160-im.width)//2,(48-im.height)//2))
        a=np.asarray(canvas,dtype=np.float32)/255.0
        a=(a-.5)/.5
        return a[None,None,:,:]

    def read_plate(self,im:Image.Image):
        logits=self.ocr.run([self.ocr_out],{self.ocr_in:self.ocr_input(im)})[0]
        z=np.asarray(logits)
        if z.ndim==3: z=z[0]
        # model exports [T,C]; tolerate [C,T]
        if z.shape[-1]!=37 and z.shape[0]==37: z=z.T
        if z.shape[-1]!=37: return "",0.0
        z=z-z.max(axis=1,keepdims=True)
        p=np.exp(z);p/=p.sum(axis=1,keepdims=True)+1e-9
        ids=p.argmax(axis=1); prev=-1; chars=[]; probs=[]
        for t,i in enumerate(ids):
            i=int(i)
            if i!=0 and i!=prev:
                chars.append(ALPHABET[i-1]); probs.append(float(p[t,i]))
            prev=i
        txt=re.sub(r"[^A-Z0-9]","","".join(chars))[:10]
        conf=float(np.exp(np.mean(np.log(np.clip(probs,1e-6,1.0))))) if probs else 0.0
        return txt,conf

    def detect_vehicles(self,im:Image.Image):
        r=self.vehicle.predict(source=np.asarray(im),imgsz=640,conf=.24,iou=.55,verbose=False)[0]
        names=r.names
        out=[]
        if r.boxes is None: return out
        for box,cf,cl in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy()):
            name=str(names[int(cl)]).lower()
            if name not in VEHICLE_CLASSES: continue
            out.append((name,float(cf),[float(x) for x in box]))
        return out

    def plate_read(self,vehicle:Image.Image):
        r=self.plate.predict(source=np.asarray(vehicle),imgsz=384,conf=.18,iou=.55,verbose=False)[0]
        if r.boxes is None or len(r.boxes)==0: return None
        cand=[]
        W,H=vehicle.size
        for box,cf in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy()):
            x1,y1,x2,y2=[float(x) for x in box]
            w,h=x2-x1,y2-y1; ar=w/max(1,h)
            if ar<1.15 or ar>8.5 or w*h<180: continue
            pad=max(3,h*.18)
            crop=vehicle.crop((max(0,int(x1-pad)),max(0,int(y1-pad)),min(W,int(x2+pad)),min(H,int(y2+pad))))
            txt,oc=self.read_plate(crop)
            if 3<=len(txt)<=10:
                cand.append((txt,float(oc)*(.72+.28*float(cf)),float(cf),crop))
        return max(cand,key=lambda x:x[1]) if cand else None

def recent_hashes(con,camera_id:str,limit=20):
    return [r[0] for r in con.execute(
      "SELECT perceptual_hash FROM camera_frames WHERE camera_id=? AND perceptual_hash IS NOT NULL ORDER BY fetched_at DESC LIMIT ?",
      (camera_id,limit)).fetchall()]

def recent_crop_hashes(con,camera_id:str,limit=100):
    return [r[0] for r in con.execute("""SELECT svo.crop_perceptual_hash
      FROM street_vehicle_observations svo JOIN camera_frames cf ON cf.frame_id=svo.frame_id
      WHERE cf.camera_id=? AND svo.crop_perceptual_hash IS NOT NULL
      ORDER BY svo.created_at DESC LIMIT ?""",(camera_id,limit)).fetchall()]

def save_frame_record(con,cam,frame_id,when,status,im=None,raw=None,error=None,vehicle_count=0):
    ph=ahash(im) if im else None
    con.execute("""INSERT OR REPLACE INTO camera_frames
      (frame_id,camera_id,camera_title,camera_view,camera_region,direction,longitude,latitude,
       image_url,fetched_at,http_status,sha256,perceptual_hash,width,height,vehicle_count,error)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(
      frame_id,cam["id"],cam["title"],cam["view"],cam["region"],cam["direction"],
      cam["longitude"],cam["latitude"],cam["href"],when,status,sha(raw) if raw else None,
      ph,im.width if im else None,im.height if im else None,vehicle_count,error
    ))
    con.commit()

def process_camera(con,models,cam,out_dir:Path,keep_frames=False):
    when=utc(); frame_id="cam:"+cam["id"]+":"+str(int(time.time()*1000))
    url=cam["href"]+("?" if "?" not in cam["href"] else "&")+"ts="+str(int(time.time()))
    try:
        raw=request(url)
        im=Image.open(io.BytesIO(raw)).convert("RGB")
    except urllib.error.HTTPError as e:
        save_frame_record(con,cam,frame_id,when,e.code,error=str(e))
        return {"camera":cam["title"],"status":"http_error","code":e.code}
    except Exception as e:
        save_frame_record(con,cam,frame_id,when,0,error=repr(e))
        return {"camera":cam["title"],"status":"error","error":repr(e)}

    ph=ahash(im)
    if any(hamming(ph,x)<=2 for x in recent_hashes(con,cam["id"])):
        save_frame_record(con,cam,frame_id,when,200,im,raw,vehicle_count=0)
        return {"camera":cam["title"],"status":"unchanged"}

    dets=models.detect_vehicles(im)
    save_frame_record(con,cam,frame_id,when,200,im,raw,vehicle_count=len(dets))
    if keep_frames:
        d=out_dir/"frames";d.mkdir(parents=True,exist_ok=True);im.save(d/(frame_id.replace(":","_")+".jpg"),quality=90)

    seen=recent_crop_hashes(con,cam["id"])
    kept=0;plates=[]
    for name,conf,(x1,y1,x2,y2) in dets:
        W,H=im.size
        pad=max(4,.04*max(x2-x1,y2-y1))
        vehicle=im.crop((max(0,int(x1-pad)),max(0,int(y1-pad)),min(W,int(x2+pad)),min(H,int(y2+pad))))
        if vehicle.width<80 or vehicle.height<50: continue
        ch=ahash(vehicle)
        if any(hamming(ch,x)<=4 for x in seen): continue
        seen.append(ch)
        blob=io.BytesIO();vehicle.save(blob,format="JPEG",quality=90);vb=blob.getvalue()
        mid="street:"+sha(vb)[:24]
        od=out_dir/"vehicles"/cam["id"];od.mkdir(parents=True,exist_ok=True)
        path=od/(mid.replace(":","_")+".jpg");path.write_bytes(vb)
        meta={"camera_id":cam["id"],"camera_title":cam["title"],"camera_view":cam["view"],
              "frame_id":frame_id,"detector_class":name,"detector_confidence":conf}
        con.execute("""INSERT OR IGNORE INTO media
          (media_id,source_id,source_page_url,image_url,local_path,jurisdiction,license,
           sha256,perceptual_hash,captured_at,imported_at,metadata_json)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",(
          mid,"tfnsw_live_cameras",cam["href"],cam["href"],str(path),"NSW",
          "Creative Commons Attribution",sha(vb),ch,when,utc(),json.dumps(meta)
        ))
        pr=models.plate_read(vehicle)
        ptxt=pconf=None
        if pr:
            ptxt,pconf,pdet,plate_crop=pr
            oid="plate:"+uuid.uuid4().hex
            con.execute("""INSERT INTO plate_observations
              (observation_id,media_id,plate_text,text_status,jurisdiction,
               detector_confidence,ocr_confidence,notes)
              VALUES(?,?,?,?,?,?,?,?)""",(
              oid,mid,ptxt,"machine_read","NSW",pdet,pconf,
              "Live street-camera OCR. Not registry truth until independently verified."
            ))
            plates.append({"text":ptxt,"confidence":round(pconf,3)})
        soid="street:"+uuid.uuid4().hex
        con.execute("""INSERT INTO street_vehicle_observations
          (street_observation_id,frame_id,media_id,detector_class,detector_confidence,
           bbox_x1,bbox_y1,bbox_x2,bbox_y2,crop_sha256,crop_perceptual_hash,
           provisional_plate,provisional_plate_confidence,created_at)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(
          soid,frame_id,mid,name,conf,x1,y1,x2,y2,sha(vb),ch,ptxt,pconf,utc()
        ))
        kept+=1
    con.commit()
    return {"camera":cam["title"],"status":"ok","detected":len(dets),"kept":kept,"plates":plates}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",type=Path,default=DEFAULT_DB)
    ap.add_argument("--out",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--camera-limit",type=int,default=12)
    ap.add_argument("--region",action="append",help="e.g. SYD_MET; repeatable")
    ap.add_argument("--interval",type=float,default=15.0)
    ap.add_argument("--cycles",type=int,default=1,help="0 = continue until interrupted")
    ap.add_argument("--vehicle-model",default="yolo26n.pt")
    ap.add_argument("--keep-full-frames",action="store_true")
    ap.add_argument("--list-cameras",action="store_true")
    args=ap.parse_args()

    idx=load_index()
    cams=cameras(idx,set(args.region) if args.region else None)
    if args.list_cameras:
        for c in cams: print(json.dumps(c,ensure_ascii=False))
        return
    con=init_db(args.db)
    models=Models(args.vehicle_model)
    args.out.mkdir(parents=True,exist_ok=True)
    cycle=0
    try:
        while args.cycles==0 or cycle<args.cycles:
            started=time.time()
            live=0
            scanned=0
            for cam in cams[:max(args.camera_limit,args.scan_limit)]:
                scanned+=1
                result=process_camera(con,models,cam,args.out,args.keep_full_frames)
                print(json.dumps(result,ensure_ascii=False),flush=True)
                if result.get("status") in {"ok","unchanged"}:
                    live+=1
                    if live>=max(1,args.camera_limit):
                        break
            print(json.dumps({"cycle":cycle+1,"scanned":scanned,"live_cameras":live,
                              "target_live_cameras":args.camera_limit},ensure_ascii=False),flush=True)
            cycle+=1
            if args.cycles and cycle>=args.cycles: break
            time.sleep(max(0,args.interval-(time.time()-started)))
    except KeyboardInterrupt:
        pass

if __name__=="__main__":
    main()
