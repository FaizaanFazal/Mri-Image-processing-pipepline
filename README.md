# SkullStrippingProject

Reproducible structural-MRI preprocessing for ADNI, AIBL, OASIS-2, and OASIS-3. The repository contains pipeline code only. Raw scans, clinical tables, manifests, processed volumes, logs, reports, and patient-derived QC images remain outside Git.

> Privacy boundary: do not commit source or processed MRI, clinical tables, dataset identifiers, acquisition dates, manifests, executed notebooks, logs, or QC images. These files may remain sensitive even when filenames are replaced.

## Pipeline

Every selected T1-weighted volume follows the same primary workflow:

```text
3D T1 sanity check
  → SynthStrip brain-mask generation on the original T1
  → mask-guided ANTs N4 bias-field correction
  → apply the SynthStrip mask in native space
  → ANTs affine registration to one MNI152 1 mm template
  → transform the binary mask with nearest-neighbor interpolation
  → brain-masked, per-volume Z-score normalization
  → automated numerical, spatial, and visual QC
```

The final modeling image is the normalized volume. Intermediate masks, N4 images, skull-stripped images, registered images, affine transforms, and status manifests are retained outside this repository to support reproducibility and interrupted-run recovery.

## Why this design is strong

The pipeline follows widely used structural-MRI preprocessing practices while keeping the primary workflow intentionally conservative:

- SynthStrip generates the mask from the original T1 instead of from an already altered derivative.
- N4 estimates its bias field inside the brain mask, reducing background influence.
- MRI intensities use linear interpolation and binary masks use nearest-neighbor interpolation.
- Affine MNI registration standardizes orientation, field of view, shape, and voxel spacing without imposing nonlinear warps in the primary pipeline.
- Z-score statistics are calculated independently for each volume from brain voxels only. No subject uses statistics from another subject or from combined train, validation, or test data.
- Background voxels are explicitly reset to zero after normalization.
- Fixed random seeds, saved parameters, software versions, template hashes, model hashes, and transforms support reproducibility.
- Atomic status-manifest checkpoints make the workflow resumable and prevent valid completed stages from being repeated unnecessarily.
- Automated QC is applied to every scan; only a small configurable number of visual examples is generated.
- Histogram matching, Nyul normalization, ComBat, denoising, and nonlinear SyN registration are deliberately excluded from the primary pipeline.

This is a best-practice-aligned baseline, not a claim that one preprocessing strategy is optimal for every scientific question. Registration choice and QC thresholds should still be validated for the intended downstream model and population.

## Repository layout

```text
SkullStrippingProject/
├── configs/                    public pipeline defaults
├── scripts/                    command-line entry points and inventory builders
├── src/skullstrip_pipeline/    conversion, preprocessing, QC, and reporting code
├── resources/                  locally downloaded third-party tools/templates
├── notebooks/                  sanitized notebooks with aggregate history
├── logs/                       local runtime logs; ignored by Git
├── reports/                    generated reports; ignored by Git
├── results/                    generated plots/QC; ignored by Git
├── requirements.txt
└── .gitignore
```

The code expects this repository to be inside a private data workspace:

```text
workspace/
├── SkullStrippingProject/
├── ADNI/                       private raw data
├── AIBL/                       private raw data
├── OASIS/                      private raw data
├── Manifests/                  private source manifests
└── Processed/
    ├── Clinical/               private organized clinical data
    └── MRI/                    private pipeline outputs
```

Paths are resolved relative to the repository parent in `src/skullstrip_pipeline/paths.py`. A different layout requires changing that module or adding configurable paths before execution.

## Setup

Python 3.12 and an NVIDIA GPU with a compatible CUDA driver are recommended. ANTsPy performs N4 and affine registration; PyTorch runs SynthStrip. CPU execution is not the tested primary configuration.

```bash
git clone <YOUR_REPOSITORY_URL>
cd SkullStrippingProject
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

PyTorch/CUDA compatibility depends on the host driver. If the pinned PyTorch wheel is unsuitable for the machine, install the correct wheel from the official PyTorch instructions and then install the remaining requirements.

### Required local resources

Third-party model weights and templates are intentionally ignored by Git. Place these files at the exact paths below:

```text
resources/synthstrip/mri_synthstrip
resources/synthstrip/synthstrip.1.pt
resources/templates/MNI152_T1_1mm_brain.nii.gz
```

SynthStrip resources can be obtained from the official FreeSurfer distribution:

```bash
mkdir -p resources/synthstrip resources/templates
curl -L https://raw.githubusercontent.com/freesurfer/freesurfer/dev/mri_synthstrip/mri_synthstrip \
  -o resources/synthstrip/mri_synthstrip
curl -L https://ftp.nmr.mgh.harvard.edu/pub/dist/freesurfer/synthstrip/models/synthstrip.1.pt \
  -o resources/synthstrip/synthstrip.1.pt
chmod +x resources/synthstrip/mri_synthstrip
```

Provide a licensed MNI152 1 mm brain template as `resources/templates/MNI152_T1_1mm_brain.nii.gz`. For an FSL installation, the standard template can typically be copied locally with:

```bash
cp "$FSLDIR/data/standard/MNI152_T1_1mm_brain.nii.gz" \
  resources/templates/MNI152_T1_1mm_brain.nii.gz
```

Do not redistribute FreeSurfer models or MNI/FSL templates until their respective license terms have been reviewed.

Verify the command-line entry point:

```bash
python scripts/run_pipeline.py --help
python scripts/run_pipeline.py skullstrip --help
```

## Running the workflow

First run a small test of three to five scans:

```bash
source .venv/bin/activate
python scripts/run_pipeline.py skullstrip \
  --dataset ADNI \
  --max-files 3 \
  --gpu-id 2 \
  --max-qc-plots 3
```

Supported dataset names are `ADNI`, `AIBL`, `OASIS2`, and `OASIS3`. Progress output reports only dataset-level counts and numeric positions; it does not print subject or image identifiers.

After reviewing the aggregate QC and private visual examples, process all pending scans:

```bash
python scripts/run_pipeline.py skullstrip \
  --dataset ADNI \
  --max-files 0 \
  --gpu-id 2 \
  --max-qc-plots 3
```

`--max-files 0` means all pending scans. A successful row with valid outputs is reused on later runs. Use `--replace` only when a pipeline change requires existing derivatives to be rebuilt.

### AIBL DICOM conversion

AIBL DICOM series must first be converted to raw 3D NIfTI volumes. Modality labels and source-to-volume provenance are retained in the private conversion manifest.

```bash
# Test conversion
python scripts/run_pipeline.py convert-aibl \
  --max-files 5 \
  --workers 1 \
  --stop-on-error

# Resume all pending conversions
python scripts/run_pipeline.py convert-aibl \
  --max-files 0 \
  --workers 2 \
  --stop-on-error
```

The AIBL preprocessing driver selects successful T1-family volumes from that conversion manifest. T2-FLAIR and SWI remain labeled but are not passed into the primary T1 pipeline.

## Outputs and checkpoints

Outputs are written outside the repository under `Processed/MRI/<DATASET>/`:

```text
source_cache/               standardized source geometry when needed
synthstrip/masks/           native binary brain masks
bias_corrected/images/      mask-guided N4 images
skullstripped/images/       N4 images with background removed
registered/images/          affine-MNI brain images
registered/masks/           nearest-neighbor MNI masks
registered/transforms/      affine transforms
normalized/images/          final brain-masked Z-score images
manifests/                  private resumable status and QC records
logs/                       commands, parameters, versions, and failures
qc_examples/                up to the configured number of private examples
```

The status manifest is written atomically after each stage. It records success/failure, completed stage, retry state, output paths, geometry, mask statistics, registration metrics, normalization metrics, and a bounded error message. It contains identifiers and must remain private.

## Automated QC

Every processed scan is checked for:

- valid 3D geometry, positive voxel sizes, finite values, and nonzero intensity range;
- binary, nonempty SynthStrip mask with matching shape/affine and explicit qform/sform;
- plausible native mask fraction, brain volume, and largest connected component;
- finite N4 output with nondegenerate brain-tissue intensities;
- zero background after native mask application;
- successful affine alignment to the fixed template;
- final template shape, 1 mm spacing, and affine agreement;
- template overlap, centroid distance, and mask connectivity;
- brain-only normalized mean near 0 and standard deviation near 1;
- exactly zero normalized background.

Registration receives one robust-initialization retry before the scan is marked as a QC failure.

### Anonymous QC report example

The following is a **fictional format example**, not a result from any participant or dataset:

| Stage | Passed | Failed | Pass rate |
|---|---:|---:|---:|
| SynthStrip brain mask | 5 | 0 | 100% |
| Mask-guided N4 | 5 | 0 | 100% |
| Native skull stripping | 5 | 0 | 100% |
| Affine MNI registration | 5 | 0 | 100% |
| Brain-masked Z-score | 5 | 0 | 100% |

| Metric | Illustrative value | Required interpretation |
|---|---:|---|
| Brain-mask volume | 1,230 mL | Must remain within configured plausibility limits |
| Template overlap | 0.94 | Must be at least 0.85 |
| Centroid distance | 4.8 mm | Must be at most 30 mm |
| Normalized brain mean | 0.0000 | Absolute value at most 0.001 |
| Normalized brain standard deviation | 1.0000 | Between 0.99 and 1.01 |
| Background maximum | 0.0000 | At most 1e-7 |

Private visual QC uses only an anonymous sample number and numeric manifest position. Its top row shows orthogonal native slices with the SynthStrip contour; the bottom row shows normalized affine-MNI slices with subject-mask and template contours. Patient-derived QC images are intentionally excluded from Git even when labels are removed.

The committed driver notebooks retain anonymous processing progress, aggregate QC tables, and aggregate QC dashboards. Embedded participant-level MRI montages are removed by `scripts/sanitize_public_notebooks.py`; the corresponding private files remain outside Git under `Processed/MRI/<DATASET>/qc_examples/`.

## Public-release checklist

Before pushing:

```bash
python scripts/sanitize_public_notebooks.py
git status --short --ignored
git ls-files | rg -i '\.(nii(\.gz)?|dcm|csv|tsv|xlsx|h5|jpg|jpeg)$' || true
git diff --cached
```

Confirm that no tracked file contains:

- subject, image, scan, accession, or site identifiers;
- raw or processed clinical values;
- acquisition dates or private source paths;
- MRI slices, masks, or patient-derived plots;
- manifests, logs, credentials, or machine-specific environment files.

`.gitignore` protects only untracked files. If sensitive content was committed previously, removing it from the working tree is insufficient; it must also be removed from Git history before publication.

## License and citation

Add a software license before publishing. A permissive license such as MIT allows reuse while requiring preservation of the copyright and license notice. Citation is a separate scholarly request, so add a `CITATION.cff` containing the final repository URL, author name, release version, and DOI when available. GitHub renders `CITATION.cff` as a “Cite this repository” action.

Do not apply the code license to ADNI, AIBL, OASIS, FreeSurfer/SynthStrip, ANTs, PyTorch, FSL/MNI templates, or any other third-party data or software. Their original terms remain controlling.
