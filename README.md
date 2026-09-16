# fetal_hc_qa

AI-assisted verification of fetal head-circumference (HC) measurements for
obstetric ultrasound QA — CST-590 Computer Science Capstone (Andie Draucker).

A diagnostic-imaging quality-assurance / second-reader pipeline: independently
re-measures fetal HC from a 2D ultrasound image, compares it to the
sonographer's recorded value, and classifies any discrepancy into an
evidence-based tier (No Flag / Review / Re-measure) using the intra-/inter-
observer reproducibility limits published in Sarris et al. (2012).

## Known, disclosed deviations (Milestone 3 build)

This sandbox had no internet access to install PyTorch or download the real
HC18 dataset, and no GPU. Two interface-compatible substitutions were made:

1. **Segmentation backend**: classical computer vision (Otsu threshold +
   morphological close + largest-contour ellipse fit) instead of a trained
   U-Net, behind the identical `HeadSegmentationModel.predict_mask()`
   interface.
2. **Dataset**: a 60-case synthetic dataset generated to exactly match the
   real HC18 Grand Challenge dataset's file/column schema, instead of the
   real dataset.

Both are fully documented in code docstrings (`segmentation_model.py`,
`scripts/generate_synthetic_dataset.py`) and in the Milestone 3 Implementation
Plan's "Known Deviations and Reconciliation Plan" section. Both are dropped
in with zero code changes elsewhere once a trained U-Net / the real HC18
dataset are available.

## Quick start

```bash
pip install numpy opencv-python pandas scipy matplotlib pytest
python scripts/generate_synthetic_dataset.py   # one-time
python scripts/run_pipeline.py                 # full batch run -> reports/
pytest tests/ -v                                # 42-test automated suite
```

See the accompanying `UserGuide_Draucker.docx` for full usage instructions,
and `Milestone3_Draucker_Final.docx` for the complete implementation writeup
(system entities, functional requirements mapping, source code listing, code
review, implementation plan, and execution evidence).

## Package layout

```
src/fetal_hc_qa/       Core package (11 modules; see Source Code Listing)
scripts/               generate_synthetic_dataset.py, run_pipeline.py
tests/                 42 pytest tests (unit + end-to-end)
data/sample_dataset/   60-case synthetic HC18-schema test dataset
reports/               Generated per-case reports + batch summary (git-ignored)
evidence/              Captured pytest/pipeline run output
```

## References

- Sarris, I., et al. (2012). Intra- and interobserver variability in fetal
  ultrasound measurements. *Ultrasound in Obstetrics & Gynecology, 39*(3), 266–273.
- van den Heuvel, T. L. A., et al. (2018). Automated measurement of fetal
  head circumference using 2D ultrasound images. *PLOS ONE, 13*(8), e0200412.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional
  networks for biomedical image segmentation. *MICCAI 2015*.
- International Fetal and Newborn Growth Consortium for the 21st Century
  (INTERGROWTH-21st). (2023). International fetal growth standards: Head
  circumference (mm) [Data table].
