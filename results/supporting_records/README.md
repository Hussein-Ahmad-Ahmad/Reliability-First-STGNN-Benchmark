# Benchmark Supporting Records

These records accompany the final article interpretation. The numerical
benchmark remains pinned to [v1.4.4](https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark/tree/v1.4.4),
commit `60c812982e530b9c743b4feeb0154310182e6639`. No training results or
historical release files are replaced by this addendum.

## Record Index

| Article location | File | Contents |
|---|---|---|
| Section VII-E, Table 16 and Table S1 | [Supporting Tables.pdf](Supporting%20Tables.pdf), [Table S1.csv](Table%20S1.csv) | All ten reference-degree controls, 30 random draws per control, source hashes and descriptive ranks |
| Section VII-C and Table S2 | [Table S2 all horizons.csv](Table%20S2%20all%20horizons.csv), [calibration_boundary_sensitivity.json](calibration_boundary_sensitivity.json) | Original and boundary-excluded PICP/MPIW for three METR-LA interval methods |
| Table 5 and auxiliary diagnostics | [checkpoint_states.json](checkpoint_states.json) | Tensor comparison of 27 auxiliary cases with archived checkpoint states |
| Appendix Tables A.4-A.6 | [RER evidence map.csv](RER%20evidence%20map.csv), [Synthesis sources.csv](Synthesis%20sources.csv), [Screening accounting.csv](Screening%20accounting.csv) | 117 final literature codes, 63-source synthesis and aggregate screening accounting |
| Figures A.1, A.3 and A.8 | [figure_generation.json](figure_generation.json), [render_selected_figures.py](render_selected_figures.py) | Input records and plotting code for the three retained production figures |
| Figure A.10 | [ig_overlap_figure.json](ig_overlap_figure.json), [render_ig_overlap.py](render_ig_overlap.py) | Seven-model top-10 sensor inclusion counts and plotting code |

[Supporting Tables.tex](Supporting%20Tables.tex) is the editable source of the
reader-facing tables. The tables retain the S1/S2 identifiers used in the article;
their availability here does not imply a separate journal supplement.

## Interpretation

S1 ranks describe responses of fixed sensor sets on epoch-100 checkpoints, not
significance tests or authenticated fidelity of the original explainer checkpoint.
All 27 inspected auxiliary states match archived epoch-100 states; 25 differ
from validation-selected states. Weight identity does not authenticate every
historical inference or establish equivalent preprocessing.

The sample-specific XAI routine uses loader-supplied inputs and native model
outputs. Target-MAE controls instead use their specified scaled-input and
inverse-output pipeline. Cross-horizon diagnostics and absolute native-output
changes are not harmonized benchmark-matched explanation fidelity. The primary
traffic assessment retains five Met and four Partial judgments, including RER-6
Partial. Model-specific attention records are not a controlled common-axis
entropy comparison; the removed entropy panel is not included in this addendum.

S2 retains the calibration origins, fitted thresholds and validity mask and
excludes the first 11 evaluation origins. The origin count changes from 3,416
to 3,405, but both evaluations contain the same 7,118,436 valid target elements
because those 11 origins contribute no valid targets. This is a boundary and
mask check, not sensitivity to removing valid observations or an exchangeability
guarantee.

The literature files are reference-anchored supporting records. They do not
recover unavailable historical search transitions. The original paired coding
passes and agreement calculation remain in `results/literature_coding/`.

## Verification

From the repository root, using Python 3.9 or later and the standard library:

```text
python results/supporting_records/verify_records.py
```

This checks file hashes, S1 source identities and arithmetic, the S2 record
consistency, checkpoint comparison summaries, and literature counts. It does not
rerun neural models or independently authenticate historical execution.

The unchanged `Evaluation-Artifacts-v1.4.4.zip` release asset supplies the frozen
JSON inputs used by the plotting and array-calculation scripts. Figure plotting
requires NumPy and Matplotlib. For example:

```text
python results/supporting_records/render_selected_figures.py --supplement PATH_TO_EVALUATION_ARCHIVE --output-dir OUTPUT_DIRECTORY --records-path OUTPUT_RECORD_JSON
python results/supporting_records/render_ig_overlap.py --supplement PATH_TO_EVALUATION_ARCHIVE --output-dir OUTPUT_DIRECTORY --records-path OUTPUT_RECORD_JSON
```

The following optional checks require authorized access to the internal files:

```text
python results/supporting_records/calibration_boundary_sensitivity.py --arrays-dir PATH_TO_ALIGNED_ARRAYS --supplement PATH_TO_EVALUATION_ARCHIVE --raw-data PATH_TO_METR_LA_DATA --output-dir OUTPUT_DIRECTORY
python results/supporting_records/check_checkpoint_states.py --archive-root ARCHIVED_CHECKPOINT_ROOT --repository-root AUXILIARY_CHECKPOINT_ROOT --supplement PATH_TO_EVALUATION_ARCHIVE --output-dir OUTPUT_DIRECTORY
```

The boundary calculation uses Python 3.12 and NumPy 2.4.4 to reproduce its
recorded floating-point results. Checkpoint comparison additionally requires
PyTorch supporting `weights_only=True`. These scripts do not train models or
execute inference. Large traffic arrays and weights are not distributed, so
these files do not provide a self-contained reproduction of all experiments.

`SHA256SUMS.txt` covers this directory except itself. The repository's
`ARTIFACT_MANIFEST.sha256` additionally covers the complete tracked release tree.
