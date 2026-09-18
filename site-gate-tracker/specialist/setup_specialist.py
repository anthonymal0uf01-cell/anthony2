from __future__ import annotations

import argparse, os, subprocess, sys, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MAMBA_REPO='https://github.com/csguoh/MambaIR.git'
MAMBA_WEIGHT_URL='https://github.com/csguoh/MambaIR/releases/download/v1.0/mambairv2_classicSR_Small_x4.pth'

def run(cmd,cwd=None,check=True):
    print('+',' '.join(map(str,cmd)),flush=True);return subprocess.run(list(map(str,cmd)),cwd=cwd,check=check)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--skip-restoration',action='store_true');args=ap.parse_args()
    py=sys.executable
    run([py,'-m','pip','install','--upgrade','pip'])
    run([py,'-m','pip','install','fastapi','uvicorn[standard]','python-multipart','opencv-python-headless','pillow','numpy','pyyaml','openocr-python==0.1.5','ultralytics','huggingface_hub'])
    # Do not silently replace the user's CUDA/PyTorch install. Install torch only if absent.
    try:import torch;print('PyTorch:',torch.__version__,'CUDA:',torch.cuda.is_available())
    except Exception:run([py,'-m','pip','install','torch','torchvision'])
    if not args.skip_restoration:
        vendor=ROOT/'.vendor'/'MambaIR';weights=ROOT/'weights'/'mambairv2_classicSR_Small_x4.pth';weights.parent.mkdir(parents=True,exist_ok=True)
        if not vendor.exists():run(['git','clone','--depth','1',MAMBA_REPO,vendor])
        if not weights.exists():
            print('Downloading MambaIRv2 Small x4 weights…');urllib.request.urlretrieve(MAMBA_WEIGHT_URL,weights)
        # MambaIR dependencies can be CUDA/platform-sensitive. Try them, but keep SVTRv2 usable if restoration deps fail.
        req=vendor/'requirements.txt'
        if req.exists():run([py,'-m','pip','install','-r',req],check=False)
        run([py,'-m','pip','install','causal-conv1d','mamba-ssm'],check=False)
        print('MambaIRv2 repo:',vendor);print('MambaIRv2 weights:',weights)
    print('\nSetup complete.')
    print('Start service with:')
    print(f'  {py} {ROOT / "advanced_anpr_service.py"}')
    print('Then on the phone tracker tap ADVANCED RECOGNISER and enter this computer LAN URL, e.g. http://192.168.1.20:8787')

if __name__=='__main__':main()
