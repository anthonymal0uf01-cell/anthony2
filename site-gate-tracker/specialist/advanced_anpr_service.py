from __future__ import annotations

import json, math, os, re, sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

# OpenOCR package/repo exposes the recognizer internals we need for raw CTC logits.
from tools.infer_rec import OpenRecognizer
from tools.infer_e2e import OpenOCRE2E

APP_DIR = Path(__file__).resolve().parent
MAMBAIR_HOME = Path(os.getenv("MAMBAIR_HOME", APP_DIR / ".vendor" / "MambaIR"))
MAMBAIR_WEIGHTS = Path(os.getenv("MAMBAIR_WEIGHTS", APP_DIR / "weights" / "mambairv2_classicSR_Small_x4.pth"))
AU_PLATE_WEIGHTS = Path(os.getenv("AU_PLATE_WEIGHTS", APP_DIR / "weights" / "au_plate_detector.pt"))
AU_OCR_SEED_WEIGHTS = Path(os.getenv("AU_OCR_SEED_WEIGHTS", APP_DIR / "weights" / "au_ocr_seed.pt"))

app = FastAPI(title="Gate Specialist ANPR v7", version="7.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

ALNUM = re.compile(r"[^A-Z0-9]")
AU_PATTERNS = [
    re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{2}$"),
    re.compile(r"^[A-Z]{3}[0-9]{2}[A-Z]$"),
    re.compile(r"^[A-Z]{3}[0-9]{3}$"),
    re.compile(r"^[0-9]{3}[A-Z]{3}$"),
    re.compile(r"^[A-Z]{2}[0-9]{3,4}$"),
    re.compile(r"^[0-9]{1,4}[A-Z]{1,4}$"),
    re.compile(r"^[A-Z0-9]{3,8}$"),  # personalised / specialist plates: deliberately broad
]


def norm_plate(s: str) -> str:
    return ALNUM.sub("", (s or "").upper())[:10]


def au_soft_prior(s: str) -> float:
    p = norm_plate(s)
    if not p:
        return 0.0
    hits = sum(bool(r.match(p)) for r in AU_PATTERNS)
    mix = 1.0 if any(c.isalpha() for c in p) and any(c.isdigit() for c in p) else 0.6
    length = 1.0 if 4 <= len(p) <= 7 else 0.65
    return min(1.0, 0.45 * min(1, hits) + 0.30 * mix + 0.25 * length)


def decode_upload(data: bytes) -> np.ndarray:
    a = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(a, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("invalid image")
    return img


def order_quad(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def rectify(img: np.ndarray, points: list[list[float]]) -> np.ndarray:
    q = order_quad(np.asarray(points))
    tl, tr, br, bl = q
    w = max(16, int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = max(8, int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def plate_box_score(item: dict[str, Any], frame_shape: tuple[int, ...]) -> float:
    text = norm_plate(item.get("transcription", ""))
    score = float(item.get("score", 0.0) or 0.0)
    pts = np.asarray(item.get("points", []), np.float32)
    if pts.size < 8:
        return -1.0
    x0, y0 = pts[:, 0].min(), pts[:, 1].min(); x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    ratio = w / h
    area_frac = (w * h) / max(1.0, frame_shape[0] * frame_shape[1])
    char_bonus = 0.35 if 3 <= len(text) <= 10 else -0.15
    shape_bonus = 0.18 if 1.3 <= ratio <= 7.5 else (0.05 if 0.75 <= ratio <= 9 else -0.1)
    size_bonus = min(0.25, math.sqrt(max(0, area_frac)) * 2.0)
    return score + char_bonus + shape_bonus + size_bonus


def quality_weight(q: dict[str, Any] | None, detector_score: float) -> float:
    q = q or {}
    sharp = float(q.get("sharpness", 0.5) or 0.5)
    glare = float(q.get("glare", 0.0) or 0.0)
    dark = float(q.get("dark", 0.0) or 0.0)
    brightness = float(q.get("brightness", 0.5) or 0.5)
    exposure = max(0.2, 1.0 - abs(brightness - 0.5) * 1.25)
    image_q = np.clip(0.30 + 0.45 * sharp + 0.15 * exposure - 0.30 * glare - 0.25 * dark, 0.08, 1.0)
    return float(np.clip(image_q * (0.45 + 0.55 * detector_score), 0.05, 1.0))


class _TinyAUOCRNet(torch.nn.Module):
    def __init__(self, nclass: int = 37) -> None:
        super().__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1,24,3,1,1), torch.nn.BatchNorm2d(24), torch.nn.SiLU(), torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(24,48,3,1,1), torch.nn.BatchNorm2d(48), torch.nn.SiLU(), torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(48,72,3,1,1), torch.nn.BatchNorm2d(72), torch.nn.SiLU(), torch.nn.MaxPool2d((2,1),(2,1)),
            torch.nn.Conv2d(72,96,3,1,1), torch.nn.BatchNorm2d(96), torch.nn.SiLU(), torch.nn.MaxPool2d((2,1),(2,1)),
            torch.nn.AdaptiveAvgPool2d((1,None)),
        )
        self.rnn = torch.nn.GRU(96,64,num_layers=1,bidirectional=True,batch_first=True)
        self.fc = torch.nn.Linear(128,nclass)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z=self.cnn(x).squeeze(2).permute(0,2,1)
        z,_=self.rnn(z)
        return self.fc(z)


class TinyAUOCRRuntime:
    def __init__(self) -> None:
        self.model = None
        self.error = None
        self.alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            if not AU_OCR_SEED_WEIGHTS.exists():
                raise FileNotFoundError("Phase-2 Australian OCR seed weights not installed")
            ckpt=torch.load(AU_OCR_SEED_WEIGHTS,map_location="cpu")
            alphabet=ckpt.get("alphabet",self.alphabet)
            if alphabet != self.alphabet:
                raise RuntimeError("unexpected Phase-2 alphabet")
            model=_TinyAUOCRNet(len(self.alphabet)+1)
            model.load_state_dict(ckpt["state_dict"],strict=True)
            model.eval().to(self.device)
            self.model=model
            self.metrics=ckpt.get("metrics",{})
        except Exception as e:
            self.error=str(e)
            self.metrics={}

    @property
    def ready(self) -> bool:
        return self.model is not None

    def _prep(self, bgr: np.ndarray) -> torch.Tensor:
        gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
        h,w=gray.shape[:2]
        scale=min(160/max(1,w),48/max(1,h))
        nw=max(1,int(round(w*scale))); nh=max(1,int(round(h*scale)))
        im=cv2.resize(gray,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
        canvas=np.full((48,160),127,dtype=np.uint8)
        x=(160-nw)//2; y=(48-nh)//2
        canvas[y:y+nh,x:x+nw]=im
        arr=(canvas.astype(np.float32)/255.0-.5)/.5
        return torch.from_numpy(arr)[None,:,:]

    @torch.inference_mode()
    def fused(self, crops: list[np.ndarray], weights: list[float]) -> tuple[str,float,float]:
        if self.model is None or not crops:
            return "",0.0,1.0
        x=torch.stack([self._prep(c) for c in crops]).to(self.device)
        logp=F.log_softmax(self.model(x).float(),dim=-1)
        w=torch.tensor(weights,dtype=logp.dtype,device=logp.device).clamp_min(1e-4)
        w=w/w.sum()
        fused=(logp*w[:,None,None]).sum(0)
        probs=fused.exp()
        ids=probs.argmax(-1).tolist()
        text=[]; prev=-1; used=[]
        for t,i in enumerate(ids):
            if i!=0 and i!=prev:
                text.append(self.alphabet[i-1]); used.append(float(probs[t,i].item()))
            prev=i
        s=norm_plate("".join(text))
        conf=float(np.prod(np.clip(used,1e-4,1.0))**(1/max(1,len(used)))) if used else 0.0
        ent=float((-(probs.clamp_min(1e-9)*probs.clamp_min(1e-9).log()).sum(-1)/math.log(probs.shape[-1])).mean().item())
        return s,conf,ent


class AUPlateDetector:
    def __init__(self) -> None:
        self.model = None
        self.error = None
        try:
            if not AU_PLATE_WEIGHTS.exists():
                raise FileNotFoundError("Australian plate detector weights not installed")
            from ultralytics import YOLO
            self.model = YOLO(str(AU_PLATE_WEIGHTS))
        except Exception as e:
            self.error = str(e)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def detect(self, bgr: np.ndarray) -> list[dict[str, Any]]:
        if self.model is None:
            return []
        result = self.model.predict(source=bgr, imgsz=960, conf=0.18, iou=0.55, verbose=False)[0]
        out=[]
        if result.boxes is None:
            return out
        xyxy=result.boxes.xyxy.detach().cpu().numpy()
        conf=result.boxes.conf.detach().cpu().numpy()
        for box,score in zip(xyxy,conf):
            x1,y1,x2,y2=[float(x) for x in box]
            if x2-x1 < 18 or y2-y1 < 7:
                continue
            out.append({"bbox":[x1,y1,x2,y2],"score":float(score)})
        return sorted(out,key=lambda x:x["score"],reverse=True)

    @staticmethod
    def crop(img: np.ndarray, box: list[float]) -> np.ndarray:
        h,w=img.shape[:2]
        x1,y1,x2,y2=box
        pw=max(2,int((x2-x1)*0.06)); ph=max(2,int((y2-y1)*0.10))
        x1=max(0,int(x1)-pw); y1=max(0,int(y1)-ph)
        x2=min(w,int(x2)+pw); y2=min(h,int(y2)+ph)
        return img[y1:y2,x1:x2].copy()


class MambaIRv2Restorer:
    def __init__(self) -> None:
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.error = None
        try:
            if not MAMBAIR_HOME.exists() or not MAMBAIR_WEIGHTS.exists():
                raise FileNotFoundError("MambaIRv2 repo/weights not installed")
            sys.path.insert(0, str(MAMBAIR_HOME))
            from basicsr.archs.mambairv2_arch import MambaIRv2
            self.model = MambaIRv2(
                upscale=4, in_chans=3, img_size=64, img_range=1.0,
                embed_dim=132, d_state=16, depths=(4,4,4,4,4,4),
                num_heads=(4,4,4,4,4,4), window_size=16, inner_rank=64,
                num_tokens=128, convffn_kernel_size=5, mlp_ratio=2.0,
                upsampler="pixelshuffle", resi_connection="1conv"
            )
            ckpt = torch.load(MAMBAIR_WEIGHTS, map_location="cpu")
            state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
            self.model.load_state_dict(state, strict=True)
            self.model.eval().to(self.device)
        except Exception as e:
            self.error = str(e)
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None

    @torch.inference_mode()
    def restore(self, bgr: np.ndarray) -> np.ndarray:
        if self.model is None:
            # Non-neural fallback keeps service operational but is never reported as MambaIRv2.
            up = cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            blur = cv2.GaussianBlur(up, (0, 0), 1.0)
            return cv2.addWeighted(up, 1.5, blur, -0.5, 0)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).permute(2,0,1).float().unsqueeze(0).to(self.device) / 255.0
        h, w = x.shape[-2:]
        ph = (16 - h % 16) % 16; pw = (16 - w % 16) % 16
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        y = self.model(x).clamp_(0, 1)
        y = y[..., :h*4, :w*4]
        out = (y[0].permute(1,2,0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


class SVTRLogitEngine:
    def __init__(self) -> None:
        self.rec = OpenRecognizer(mode="server", backend="torch", use_gpu="auto")
        self.device = self.rec.device

    def _batch(self, images: list[np.ndarray]) -> torch.Tensor:
        batch_data = []
        max_w = max_h = 0
        for img in images:
            data = {"image": img}
            batch = self.rec.transform(data, self.rec.ops[1:])
            arr = batch[0] if isinstance(batch[0], np.ndarray) else batch[0].numpy()
            h, w = arr.shape[-2:]; max_h = max(max_h, h); max_w = max(max_w, w)
            batch_data.append(arr)
        padded = np.zeros((len(batch_data), 3, max_h, max_w), np.float32)
        for i, arr in enumerate(batch_data):
            h, w = arr.shape[-2:]; padded[i, :, :h, :w] = arr
        return torch.from_numpy(padded).to(self.device)

    @torch.inference_mode()
    def logits(self, images: list[np.ndarray]) -> torch.Tensor:
        x = self._batch(images)
        pred = self.rec.model(x, None)
        if isinstance(pred, dict):
            pred = next(v for v in pred.values() if torch.is_tensor(v))
        elif isinstance(pred, (list, tuple)):
            pred = next(v for v in pred if torch.is_tensor(v))
        if pred.ndim != 3:
            raise RuntimeError(f"unexpected SVTR output shape: {tuple(pred.shape)}")
        return F.log_softmax(pred.float(), dim=-1)

    def fuse(self, logits: torch.Tensor, weights: list[float]) -> torch.Tensor:
        w = torch.tensor(weights, dtype=logits.dtype, device=logits.device).clamp_min(1e-4)
        w = w / w.sum()
        return (logits * w[:, None, None]).sum(dim=0)

    def decode(self, fused_log_probs: torch.Tensor) -> tuple[str, float]:
        # OpenOCR's decoder can consume a one-item batch of raw scores/probabilities.
        result = self.rec.post_process_class(fused_log_probs.unsqueeze(0), torch_tensor=True)[0]
        return norm_plate(result[0]), float(result[1])


au_seed = TinyAUOCRRuntime()
plate_detector = AUPlateDetector()
restorer = MambaIRv2Restorer()
svtr = SVTRLogitEngine()
e2e = OpenOCRE2E(mode="mobile", backend="onnx", drop_score=0.10, det_box_type="quad", use_gpu="auto")


def align_time(a: torch.Tensor, target_t: int) -> torch.Tensor:
    if a.shape[0] == target_t:
        return a
    return F.interpolate(a.transpose(0,1).unsqueeze(0), size=target_t, mode="linear", align_corners=False)[0].transpose(0,1)


def uncertainty(logp: torch.Tensor) -> tuple[float, float]:
    p = logp.exp().clamp_min(1e-9)
    ent = (-(p * p.log()).sum(-1) / math.log(max(2, p.shape[-1]))).mean().item()
    top2 = torch.topk(p, 2, dim=-1).values
    margin = (top2[:,0] - top2[:,1]).mean().item()
    return float(ent), float(margin)


def _edit_distance(a: str, b: str) -> int:
    a,b=norm_plate(a),norm_plate(b)
    d=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nd=[i]
        for j,y in enumerate(b,1):
            nd.append(min(nd[-1]+1,d[j]+1,d[j-1]+(x!=y)))
        d=nd
    return d[-1]


def arbitrate_recognizers(
    svtr_text: str, svtr_conf: float, svtr_entropy: float,
    au_text: str, au_conf: float, au_entropy: float,
) -> dict[str, Any]:
    s,a=norm_plate(svtr_text),norm_plate(au_text)
    sp=au_soft_prior(s); ap=au_soft_prior(a)
    # Agreement between independently trained branches is the strongest evidence.
    if s and a and s==a:
        conf=float(np.clip(0.58*svtr_conf+0.42*au_conf+0.08,0,0.995))
        return {"text":s,"confidence":conf,"agreement":True,"source":"AU+SVTR","disagreement":False}

    # One-character disagreements remain unresolved unless one branch is clearly stronger.
    dist=_edit_distance(s,a) if s and a else 99
    s_score=svtr_conf*(0.94+0.06*sp)*(1.0-0.28*min(1.0,svtr_entropy))
    a_score=au_conf*(0.90+0.10*ap)*(1.0-0.34*min(1.0,au_entropy))

    if a and a_score>=0.88 and a_score>=s_score+0.12:
        return {"text":a,"confidence":float(min(.985,a_score)),"agreement":False,"source":"AU","disagreement":bool(s)}
    if s and s_score>=0.88 and s_score>=a_score+0.12:
        return {"text":s,"confidence":float(min(.985,s_score)),"agreement":False,"source":"SVTR","disagreement":bool(a)}

    # A close disagreement is valuable evidence, but not enough to invent certainty.
    cand=a if a_score>s_score else s
    return {"text":cand,"confidence":float(max(a_score,s_score)),"agreement":False,"source":"CONFLICT","disagreement":bool(s and a),"edit_distance":dist}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "svtrv2": True,
        "au_ocr_seed": au_seed.ready,
        "au_ocr_seed_error": au_seed.error,
        "au_ocr_seed_metrics": au_seed.metrics,
        "au_plate_detector": plate_detector.ready,
        "au_plate_detector_error": plate_detector.error,
        "mambairv2": restorer.ready,
        "mambairv2_error": restorer.error,
        "device": str(svtr.device),
    }


@app.post("/recognize")
async def recognize(frames: list[UploadFile] = File(...), qualities: str = Form("[]")) -> dict[str, Any]:
    qlist = json.loads(qualities or "[]")
    raw_frames = [decode_upload(await f.read()) for f in frames[:12]]
    if len(raw_frames) < 2:
        return {"text":"", "confidence":0.0, "entropy":1.0, "status":"UNRESOLVED", "reason":"need multiple frames"}

    # 1) locate the plate in each frame. Prefer the Australian detector trained
    # on Australian plate boxes. Generic OCR text detection is fallback only.
    crops, weights, detector_text = [], [], []
    if plate_detector.ready:
        for i, img in enumerate(raw_frames):
            items=plate_detector.detect(img)
            if not items:
                continue
            best=items[0]
            crop=plate_detector.crop(img,best["bbox"])
            if crop.shape[0] < 8 or crop.shape[1] < 20:
                continue
            crops.append(crop)
            weights.append(quality_weight(qlist[i] if i < len(qlist) else None,float(best["score"])))
            detector_text.append("")
    else:
        detected, _ = e2e(img_numpy_list=raw_frames, is_visualize=False, crop_infer=True, rec_batch_num=4)
        for i, (img, items) in enumerate(zip(raw_frames, detected or [])):
            if not items:
                continue
            ranked = sorted(items, key=lambda it: plate_box_score(it, img.shape), reverse=True)
            best = ranked[0]
            if plate_box_score(best, img.shape) < 0.15:
                continue
            crop = rectify(img, best["points"])
            if crop.shape[0] < 8 or crop.shape[1] < 20:
                continue
            crops.append(crop)
            weights.append(quality_weight(qlist[i] if i < len(qlist) else None, float(best.get("score", 0.5))))
            detector_text.append(norm_plate(best.get("transcription", "")))

    if len(crops) < 2:
        return {"text":"", "confidence":0.0, "entropy":1.0, "status":"UNRESOLVED", "reason":"plate region not stable across frames", "detector_text":detector_text}

    # Keep a diverse but bounded temporal evidence set.
    order = np.argsort(np.asarray(weights))[::-1][:8]
    crops = [crops[i] for i in order]; weights = [weights[i] for i in order]

    # Phase-2 Australian branch: the real trained NSW/NHV checkpoint fuses
    # raw CTC probabilities across the selected frames before decoding.
    au_seed_text, au_seed_conf, au_seed_entropy = au_seed.fused(crops, weights) if au_seed.ready else ("",0.0,1.0)

    # 2) direct SVTRv2 branch: raw CTC log-probabilities, fused before decoding.
    direct_logits = svtr.logits(crops)
    direct_fused = svtr.fuse(direct_logits, weights)

    # 3) hard-case restoration branch. Uses actual MambaIRv2 when installed; otherwise a marked fallback.
    restored = [restorer.restore(c) for c in crops]
    restored_logits = svtr.logits(restored)
    restored_fused = svtr.fuse(restored_logits, weights)

    # 4) align time axis then ensemble in probability space BEFORE any string is decoded.
    target_t = max(direct_fused.shape[0], restored_fused.shape[0])
    d = align_time(direct_fused, target_t)
    r = align_time(restored_fused, target_t)
    # Direct evidence dominates; restoration contributes more on genuinely poor imagery.
    mean_q = float(np.mean(weights))
    restore_alpha = float(np.clip(0.22 + (0.55 - mean_q) * 0.45, 0.18, 0.42))
    fused = torch.logaddexp(d + math.log(1.0 - restore_alpha), r + math.log(restore_alpha))

    text, rec_conf = svtr.decode(fused)
    ent, margin = uncertainty(fused)
    prior = au_soft_prior(text)
    # AU structure is only a soft confidence prior. It never rewrites the decoded string.
    calibrated = float(np.clip(rec_conf * (0.91 + 0.09 * prior) * (0.72 + 0.28 * margin), 0, 0.999))
    decision = arbitrate_recognizers(text, calibrated, ent, au_seed_text, au_seed_conf, au_seed_entropy)
    final_text = decision["text"]
    final_conf = float(decision["confidence"])
    # Agreement can confirm at a lower branch-level threshold. A single branch must be
    # exceptionally strong. Unresolved disagreement never gets silently promoted.
    confirmed = (
        len(crops) >= 3 and (
            (decision["agreement"] and final_conf >= 0.80)
            or (decision["source"] == "AU" and final_conf >= 0.90 and au_seed_entropy <= 0.48)
            or (decision["source"] == "SVTR" and final_conf >= 0.90 and ent <= 0.60)
        )
    )

    return {
        "text": final_text,
        "confidence": final_conf,
        "decision_source": decision["source"],
        "recognizer_agreement": decision["agreement"],
        "recognizer_disagreement": decision["disagreement"],
        "au_seed_text": au_seed_text,
        "au_seed_confidence": au_seed_conf,
        "au_seed_entropy": au_seed_entropy,
        "raw_recognition_confidence": rec_conf,
        "entropy": ent,
        "character_margin": margin,
        "au_soft_prior": prior,
        "status": "CONFIRMED" if confirmed else "UNRESOLVED",
        "frames_used": len(crops),
        "restoration": "MambaIRv2-Small-x4" if restorer.ready else "fallback-unsharp",
        "restore_alpha": restore_alpha,
        "detector_text": detector_text,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8787")))
