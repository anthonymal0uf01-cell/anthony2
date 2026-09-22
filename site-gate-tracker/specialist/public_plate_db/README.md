# Australian Public Plate Corpus

This directory defines a licence-auditable SQLite corpus for Australian registration-plate and vehicle research.

## What is public now

The initial catalogue includes:

- **2,720** Australian multi-state plate images from Roboflow `australian-license-plates` (CC BY 4.0).
- **456** NSW-white plate images from Roboflow `AU-NSW-WHITE` (CC BY 4.0).
- **210** additional Australian plate images from `Licence Plates Australia` (CC BY 4.0).
- Wikimedia Commons Australian plate and vehicle categories, imported with **per-file author/licence metadata**.
- Live Traffic NSW as a **current vehicle-appearance** source, not automatically treated as verified plate-text ground truth.
- TfNSW registration statistics as fleet/configuration priors; those datasets are aggregate statistics, not a public list of individual registration numbers.

The three Roboflow sources total **3,386 listed Australian images before deduplication**.

## Label discipline

Every plate text has a status:

- `none` — no text label.
- `machine_read` — OCR output only.
- `title_inferred` — inferred from a public filename/description.
- `human_verified` — manually checked against the image.
- `registry_verified` — confirmed by an authorised vehicle-data source.
- `synthetic_exact` — generated plate whose text is known exactly.

Only the last three should normally be treated as hard OCR ground truth.

## Build

```bash
python build_public_plate_db.py --stats
```

Harvest a Commons category:

```bash
python build_public_plate_db.py \
  --commons-category "Automobiles with license plates of New South Wales" \
  --source-id commons_au_automobiles \
  --stats
```

Import a Roboflow YOLO export after downloading it under its licence:

```bash
python build_public_plate_db.py \
  --roboflow-root /path/to/australian-license-plates \
  --source-id roboflow_au_multistate \
  --stats
```

## Database direction

The corpus intentionally separates:

```
image/media
    ↓
plate observation
    ↓
plate text + confidence + verification status
    ↓
vehicle identity (when legitimately resolved)
    ↓
VIN / make / model / variant / vehicle class / specifications
```

It contains no owner/person fields. Plate/VIN enrichment should use an authorised data source and record the evidence source and applicable usage terms.
