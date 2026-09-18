#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path

def norm(s): return "".join(c for c in (s or "").upper() if c.isalnum())

def lev(a,b):
    a,b=norm(a),norm(b);d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        n=[i]
        for j,y in enumerate(b,1): n.append(min(n[-1]+1,d[j]+1,d[j-1]+(x!=y)))
        d=n
    return d[-1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("predictions",type=Path,help="CSV/JSONL fields: truth,pred,confidence,status,condition")
    args=ap.parse_args()
    rows=[]
    if args.predictions.suffix.lower()==".jsonl":
        rows=[json.loads(x) for x in args.predictions.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        with args.predictions.open(encoding="utf-8",newline="") as f: rows=list(csv.DictReader(f))
    n=len(rows);exact=0;chars=0;edits=0;resolved=0;brier=0;by={}
    for r in rows:
        t,p=norm(r.get("truth","")),norm(r.get("pred",""))
        ok=t==p and bool(t);exact+=ok;chars+=max(1,len(t));edits+=lev(t,p)
        status=(r.get("status") or "CONFIRMED").upper()
        resolved+=status!="UNRESOLVED"
        try:c=float(r.get("confidence",0) or 0)
        except:c=0
        brier+=(c-(1 if ok else 0))**2
        cond=r.get("condition") or "all"
        q=by.setdefault(cond,[0,0])
        q[0]+=1;q[1]+=int(ok)
    out={"samples":n,"exact_match":exact/n if n else 0,"character_error_rate":edits/chars if chars else 0,
         "resolved_rate":resolved/n if n else 0,"brier":brier/n if n else 0,
         "by_condition":{k:{"n":v[0],"exact_match":v[1]/v[0]} for k,v in by.items()}}
    print(json.dumps(out,indent=2))
if __name__=="__main__":main()
