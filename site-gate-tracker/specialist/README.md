# Site Gate Tracker v7 — Australia Specialist

## Objective

Maintain the most probable **identity and operational state of each physical vehicle through time**. A number plate is an evidence channel, not the vehicle itself.

The production architecture is now split into four systems:

1. **Phone identity layer** — local vehicle tracking, gate state, prime/trailer grouping, IN/OUT ledger.
2. **Australian perception layer** — Australian plate detector + current Australian vehicle taxonomy/priors.
3. **Advanced recognition layer** — multi-frame SVTRv2 raw-logit fusion + hard-case MambaIRv2 restoration + uncertainty gating.
4. **Learning layer** — difficult site passes are retained, corrected and used for the final domain-adaptation stage.

## Current runtime

Open `site-gate-tracker/v7.html`.

The phone tracker now:

- maintains persistent vehicle tracks;
- associates heavy components over multiple frames before treating them as one truck-and-dog group;
- suppresses the second gate crossing for a grouped prime/trailer without discarding the group identity;
- keeps primary and secondary plate evidence separately;
- writes grouped vehicle IDs and both plates into the movement ledger;
- retains difficult/partial cases locally for specialist learning;
- treats a truck-and-dog with only one readable plate as `PARTIAL` so the missing trailer/prime plate continues collecting evidence.

It remains an operations tally aid, **not a safety-critical traffic-control system**.

## Australia-specialist data stack

See:

- `AU_DATA_CATALOG_2026.json` — source, role, licence and production/research policy.
- `AUSTRALIA_SPECIALIST_2026.md` — complete curriculum.
- `au_taxonomy.py` — Australian visual/operational classes and truck-and-dog hypotheses.

Production data is deliberately separated from research-only data. For example, Mixed Signals V2X is useful for research but is CC BY-NC-SA and is excluded from commercial production training.

## One-command preparation

```bash
python site-gate-tracker/specialist/setup_specialist.py

python site-gate-tracker/specialist/run_au_pipeline.py \
  --work au_specialist_work \
  --synthetic-count 250000
```

This prepares current Australian official priors plus synthetic Australian OCR data.

With a Roboflow key:

```bash
export ROBOFLOW_API_KEY=...
python site-gate-tracker/specialist/run_au_pipeline.py \
  --work au_specialist_work \
  --roboflow
```

To train when GPU compute is available:

```bash
python site-gate-tracker/specialist/run_au_pipeline.py \
  --work au_specialist_work \
  --roboflow \
  --train-detector \
  --train-ocr
```

With exported corrected gate cases, add:

```bash
--gate-zip gate-specialist-YYYY-MM-DD.zip
```

The OCR curriculum is:

```
official SVTRv2 checkpoint
        ↓
Australian synthetic NSW/NHV plates
        ↓
corrected real physical-gate plates
        ↓
gate specialist
```

## Advanced recogniser

`advanced_anpr_service.py` now prefers `weights/au_plate_detector.pt` when present.

Recognition path:

```
Australian plate detector
        ↓
multi-frame plate crops
        ├── direct SVTRv2
        └── MambaIRv2 restoration → SVTRv2
                    ↓
        raw CTC log-probability fusion
                    ↓
          uncertainty + soft AU prior
                    ↓
             identity evidence
```

If the Australian detector is not installed yet, plate localisation falls back to the generic OpenOCR path. If MambaIRv2 is unavailable, the service reports that fact and uses a non-neural enhancement fallback rather than claiming Mamba restoration occurred.

Check:

```
http://<computer>:8787/health
```

The health response exposes Australian detector readiness, SVTRv2 readiness and MambaIRv2 readiness.

## Site learning

v7 retains hard/uncertain passes in IndexedDB, including darkness, glare, blur and exposure metadata. It rejects near-duplicate frames and keeps a bounded evidence set.

For truck-and-dog groups, the learning export keeps:

- primary candidate / confirmed plate;
- secondary candidate / confirmed plate;
- separate label sources;
- image-quality metadata;
- advanced-recogniser result when available.

`prepare_gate_dataset.py` now extracts **both prime and trailer labels** into independent recognition crops.

## Model promotion

Do not promote new weights based only on training loss. Required metrics are:

- exact plate match;
- character error rate;
- unresolved rate;
- confidence calibration/Brier score;
- performance by night/glare/blur/oblique/dirt condition;
- vehicle duplicate-event rate;
- truck-and-dog grouping error;
- IN/OUT direction error;
- manual corrections per 100 passes.

Use `evaluate_au_specialist.py` for the OCR metrics and a reviewed site shift for operational metrics.

## Current execution status — 18 September 2026

**Complete and CI-tested:**

- Australian source/licence catalog
- current Australian vehicle/heavy-vehicle taxonomy
- official-data bootstrap
- Australian fleet/configuration prior builder
- NSW/NHV synthetic OCR generator
- perceptual dataset deduplication/merge
- YOLO26 Australian plate-detector trainer
- staged SVTRv2 Australian → gate curriculum
- multi-frame raw-logit recogniser
- MambaIRv2 hard-case branch
- Australian detector runtime preference
- prime/trailer grouping and two-plate ledger
- two-plate hard-case collection and training export
- evaluation harness
- one-command orchestrator
- GitHub smoke tests

**Not falsely marked complete:**

- No production Australian detector/SVTRv2 checkpoint has been trained in this session.
- The connected Hugging Face Jobs endpoint returned HTTP 402 Payment Required when compute was requested.
- Roboflow training exports require a valid `ROBOFLOW_API_KEY`.
- The final gate-specialist stage requires real confirmed/corrected site passes.

Those are execution/data inputs, not missing architecture.
