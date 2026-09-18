# Hugging Face curriculum for Site Gate Tracker v7 Specialist

The model should **not** be trained by concatenating every license-plate dataset. The datasets are deliberately separated by what they are allowed to teach.

## Curriculum

1. **Plate localisation**
   - `justjuu/license-plate-detection` — CC BY 4.0, 8,823 rows. General plate boxes.
   - `evan6007/TLPD` — MIT, ~3,032 images. Polygon geometry; useful for oblique plates.

2. **Adverse-condition robustness**
   - `thasan3003/NightLPBD_Nighttime_License_Plates_of_Bangladesh` — CC BY 4.0.
   - Use it for low-light, headlights, glare, blur, exposure and localisation.
   - Do **not** use its Bangla plate strings to teach NSW plate grammar.

3. **Multi-frame / low-resolution recognition**
   - `kv1388/FANVID-Face_and_License_Plate_Recognition_in_Low-Resolution_Videos` — CC BY 4.0.
   - Its license-plate task is explicitly designed around plates that may be unreadable in one frame and benefit from temporal evidence.

4. **Generic OCR warm-up**
   - `prithivMLmods/Number-Plate-Recognition` — Apache 2.0, small labelled set.
   - Use as a weak Latin-alphanumeric OCR seed only.

5. **Australian/domain adaptation**
   - Prefer genuinely Australian examples with compatible licences.
   - Do not make paid-preview / NC-ND datasets part of the automatic pipeline.
   - Australian format knowledge is a **soft prior**, not a source of ground truth.

6. **Gate specialist — highest priority**
   - v7 keeps difficult local passes: night, dirt, glare, oblique angle, motion blur, trailers and unresolved plates.
   - Confirmed/manual corrections convert those samples into labelled data.
   - Fine-tune last on this set and oversample it so generic foreign datasets cannot dominate the final model.

## Weighting target

A useful first schedule:

- General detector data: 1.0x
- Adverse-condition data: 1.5x
- Multi-frame low-resolution data: 2.0x
- Australian data: 3.0x
- Corrected gate data: 5.0x

As the gate set grows, progressively reduce generic OCR data. The end state is a model whose broad visual perception came from public datasets but whose identity decisions are dominated by the physical gate it operates at.

## Bootstrap

```bash
pip install huggingface_hub
python site-gate-tracker/specialist/bootstrap_hf_data.py --out data/hf
```

This downloads each source into its own directory and writes `sources.json` with role and licence metadata. Keeping sources separate is intentional: it lets training code decide what each dataset can influence instead of blindly mixing plate syntax.
