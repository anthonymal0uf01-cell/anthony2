# Site Gate Tracker v7 — Gate Specialist

v7 adds a learning system on top of the v6 vehicle tracker.

## Objective

Maintain the most probable identity and state of each physical vehicle through time. A number plate is one evidence channel, not the vehicle identity itself.

## Phone layer

Open `site-gate-tracker/v7.html`.

The phone still performs the normal vehicle tracking / gate tally. v7 additionally stores difficult or uncertain passes in IndexedDB. It measures darkness, glare, sharpness and exposure, rejects near-duplicate frames, and retains a bounded diverse evidence set per track.

When the tracker later confirms a plate, that label is propagated to that track's stored difficult frames. If it remains unresolved, use **LABEL LATEST HARD CASE** to correct it. **EXPORT TRAINING ZIP** creates images plus `manifest.jsonl`.

The hard-case buffer remains on the phone unless explicitly exported or an Advanced Recogniser URL is configured.

## Advanced multi-frame recogniser

`advanced_anpr_service.py` provides:

- OpenOCR SVTRv2 recognition on raw model outputs.
- Multi-frame quality weighting.
- Temporal fusion at CTC log-probability level before decoding.
- A direct recognition path plus MambaIRv2 Small x4 restoration path.
- Entropy and character-margin uncertainty gating.
- Australian/NSW-style plate structure as a soft confidence prior only; it never rewrites the decoded string.
- Fallback image enhancement if MambaIRv2 dependencies are unavailable, while keeping the service operational.

Install:

```bash
python site-gate-tracker/specialist/setup_specialist.py
```

Run:

```bash
python site-gate-tracker/specialist/advanced_anpr_service.py
```

On the phone, tap **ADVANCED RECOGNISER** and enter the computer's LAN URL, for example `http://192.168.1.20:8787`.

Check `http://<computer>:8787/health` to see whether SVTRv2 and MambaIRv2 are active.

## Gate-specific training loop

1. Run the tracker during normal work.
2. Difficult cases are retained automatically.
3. Confirm or correct plate identities.
4. Export the training ZIP.
5. Prepare recognition crops:

```bash
python site-gate-tracker/specialist/prepare_gate_dataset.py gate-specialist-YYYY-MM-DD.zip --out gate_dataset
```

6. Fine-tune SVTRv2:

```bash
python site-gate-tracker/specialist/train_gate_specialist.py gate_dataset --epochs 25
```

The training script clones OpenOCR if required, starts from the official SVTRv2 server checkpoint, points the train/eval configuration at the gate dataset and produces a gate-specialist checkpoint. It also prints the ONNX export command for deployment.

## Learning direction

The dataset should increasingly contain the conditions the generic model gets wrong: night work, rain, dirt, headlights/glare, dog trailers, unusual NSW plates, oblique views, compression and motion blur. The intent is to reduce dependence on unrelated foreign plate distributions and specialise the recognition system to this gate and this traffic.
