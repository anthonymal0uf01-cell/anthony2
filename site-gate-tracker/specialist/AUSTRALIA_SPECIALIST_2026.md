# Australia Specialist 2026 — production training stack

Current as of **18 September 2026**.

## Production rule

Public data is not automatically synonymous with production-safe training data. The repository now separates sources into:

- **commercial_ok** — may enter production training subject to attribution/terms.
- **commercial_ok_with_plate_mask** — government traffic imagery can teach Australian vehicle/context appearance only after plate regions are masked.
- **research_only** — useful for architecture, benchmarking and experiments but excluded from commercial production weights.
- **reference_only** — standards, taxonomies and historical pattern references; never copied into model weights as image data.

See `AU_DATA_CATALOG_2026.json`.

## Production curriculum

### 1. Australian plate detector

Download/fork the CC BY 4.0 Australian detection sets, then merge them with:

```bash
python merge_au_plate_datasets.py /data/nsw-white /data/au-multistate --out data/au_plate_merged
python train_au_plate_detector.py data/au_plate_merged
```

The merge step normalises every plate class to `license_plate` and perceptually deduplicates images. This prevents the 456 NSW subset being accidentally double-counted when it also appears in a combined corpus.

### 2. Australian OCR warm-up

Generate large amounts of synthetic NSW and National Heavy Vehicle plates:

```bash
python generate_au_synthetic_plates.py --count 250000 --out data/au_synth
```

The generator randomises common NSW layout patterns, standard yellow/black and black/white appearances, NHV white/black + blue sash, plate aspect ratios, perspective, blur, low resolution, darkness, glare, dirt and JPEG quality.

Synthetic text is **not** used as a hard runtime grammar. It exists to teach the recogniser Australian-looking glyph/layout distributions.

### 3. Current Australian priors

```bash
python bootstrap_au_data.py --out data/australia
python build_au_priors.py data/australia --out data/au_priors.json
```

The priors come from current government statistics: NSW heavy-vehicle configurations, Victorian Austroads-class traffic counts and federal road-vehicle statistics. They influence sampling and identity hypotheses, never overwrite camera evidence.

### 4. SVTRv2 curriculum

```bash
python train_au_curriculum.py --synthetic data/au_synth --output runs/au_rec
```

When corrected gate examples exist:

```bash
python prepare_gate_dataset.py gate-specialist-export.zip --out data/gate
python train_au_curriculum.py --synthetic data/au_synth --gate data/gate --output runs/au_rec
```

Stage order is deliberately **Australian synthetic → corrected physical gate**. The final site stage therefore dominates generic visual knowledge.

### 5. Restricted research lane

Mixed Signals V2X (Sydney), LRLPR-26 and NARRATE are useful research datasets but are not automatically mixed into production training. Mixed Signals is CC BY-NC-SA; LRLPR-26 requires non-commercial institutional access; NARRATE access terms must be verified.

Their value is architecture/evaluation: temporal association, low-resolution fusion, uncertainty, multimodal context and hard-case benchmarks.

## Australian vehicle identity

The runtime should not force a one-frame "truck" label. It maintains:

```
physical vehicle identity
  ├─ visual class
  ├─ trajectory
  ├─ prime-mover component
  ├─ trailer/dog component(s)
  ├─ plate evidence per component
  ├─ axle/geometry hypotheses
  └─ operational state: approach / waiting / entering / inside / exiting
```

`au_taxonomy.py` maps visual classes to Austroads/NHVR operational hypotheses. A current 3-axle rigid + 4-axle dog is represented as a 7-axle truck-and-dog hypothesis, but the label only becomes strong after multi-frame geometry/association evidence.

## Evaluation gates

Do not promote a new model on training loss. Promotion requires:

- plate exact-match rate
- character error rate
- unresolved rate
- confidence calibration (Brier score)
- exact match by hard condition (night/glare/blur/oblique/dirt)
- vehicle duplicate-event rate
- truck-and-dog grouping error
- IN/OUT direction error
- manual corrections per 100 passes

Use `evaluate_au_specialist.py` for OCR metrics; vehicle/event metrics come from the site ledger and a manually reviewed validation shift.

## What remains inherently site-dependent

No public dataset can pre-supply the exact camera height, gate geometry, headlight angle, dust, truck fleet or dog-trailer appearance at the physical work site. v7 therefore keeps the hard-case learning buffer as the final adaptation layer. Public Australian data reduces the cold-start problem; site data finishes the model.
