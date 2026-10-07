#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIVER_ROOT = PROJECT_ROOT / "notebooks" / "drivers"

DRIVERS = {
    "04_adni_driver.ipynb": {
        "dataset": "ADNI", "label": "adni", "max_files": "None", "gpu": 1,
        "run_preprocessing": True,
        "title": "# adni full preprocessing driver",
        "description": "this driver runs the shared preprocessing and saves anonymous aggregate qc information plus no more than three visual qc examples.",
    },
    "05_aibl_driver.ipynb": {
        "dataset": "AIBL", "label": "aibl", "max_files": "None", "gpu": 2,
        "run_preprocessing": True,
        "title": "# aibl full preprocessing driver",
        "description": "this driver uses qc-passed raw t1 3d nifti files and saves anonymous aggregate qc information plus no more than three visual qc examples.",
    },
    "06_oasis2_driver.ipynb": {
        "dataset": "OASIS2", "label": "oasis2", "max_files": "None", "gpu": 2,
        "run_preprocessing": True,
        "title": "# oasis2 full preprocessing driver",
        "description": "this driver runs the shared preprocessing and saves anonymous aggregate qc information plus no more than three visual qc examples.",
    },
    "07_oasis3_driver.ipynb": {
        "dataset": "OASIS3", "label": "oasis3", "max_files": "None", "gpu": 1,
        "run_preprocessing": False,
        "title": "# oasis3 full preprocessing driver",
        "description": "this driver runs the shared preprocessing and saves anonymous aggregate qc information plus no more than three visual qc examples.",
    },
}


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip() + "\n")


def markdown(text: str):
    return nbformat.v4.new_markdown_cell(text.strip() + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("--only", choices=[settings["dataset"] for settings in DRIVERS.values()])
args = parser.parse_args()

for filename, settings in DRIVERS.items():
    if args.only and settings["dataset"] != args.only:
        continue
    path = DRIVER_ROOT / filename
    notebook = nbformat.read(path, as_version=4)
    dataset = settings["dataset"]
    label = settings["label"]
    gpu = settings["gpu"]
    notebook.cells = [
        markdown(f"{settings['title']}\n\n{settings['description']}"),
        code(f"""
from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, display

candidates = [Path.cwd(), *Path.cwd().parents, Path.cwd() / 'SkullStrippingProject']
PROJECT_ROOT = next(path for path in candidates if (path / 'src' / 'skullstrip_pipeline').exists())
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from skullstrip_pipeline.driver import run_dataset
from skullstrip_pipeline.reporting import build_dataset_qc_report, save_anonymous_qc_examples, save_dataset_qc_summary_plot

DATASET = '{dataset}'
MAX_FILES = {settings['max_files']}
REPLACE = False
SAVE_QC_PLOTS = True
MAX_QC_PLOTS = 3
GPU_PHYSICAL_ID = {gpu}
RUN_PREPROCESSING = {settings['run_preprocessing']}

print(f'{{DATASET.lower()}} settings: run preprocessing={{RUN_PREPROCESSING}}, max files={{MAX_FILES}}, replace={{REPLACE}}, gpu={{GPU_PHYSICAL_ID}}')
"""),
        markdown("## preprocessing"),
        code("""
if RUN_PREPROCESSING:
    status = run_dataset(
        DATASET,
        max_files=MAX_FILES,
        replace=REPLACE,
        save_qc_plots=SAVE_QC_PLOTS,
        max_qc_plots=MAX_QC_PLOTS,
        gpu_physical_id=GPU_PHYSICAL_ID,
        progress=True,
    )
else:
    manifest_path = PROJECT_ROOT.parent / 'Processed' / 'MRI' / DATASET / 'manifests' / f'{DATASET.lower()}_preprocessing_manifest.csv'
    status = pd.read_csv(manifest_path, low_memory=False)
    print(f'{DATASET.lower()} report-only mode: loaded {len(status)} manifest rows without reprocessing')
"""),
        markdown("## qc report and sanity checks"),
        code("""
qc_report = build_dataset_qc_report(status)

print('manifest status')
display(qc_report['status'])
print('stage qc summary')
display(qc_report['stage_qc'])
print('final-output sanity checks')
display(qc_report['sanity_checks'])
print('qc metric distribution for successful scans')
display(qc_report['metric_summary'])
print('final spatial geometry')
display(qc_report['geometry'])

if not qc_report['failures'].empty:
    print('anonymous failure summary')
    display(qc_report['failures'])
else:
    print('anonymous failure summary: no failures')
"""),
        markdown("## saved qc plots"),
        code("""
summary_plot = save_dataset_qc_summary_plot(status, DATASET)
visual_selection = save_anonymous_qc_examples(status, DATASET, MAX_QC_PLOTS)

print(f'aggregate qc plot saved: {summary_plot.relative_to(PROJECT_ROOT.parent)}')
print(f'anonymous visual qc plots saved: {len(visual_selection)} of maximum {MAX_QC_PLOTS}')
print('list_position is the 1-based row number in the dataset preprocessing manifest')
display(visual_selection)
display(Image(filename=str(summary_plot)))
for plot_path in visual_selection['plot_path']:
    display(Image(filename=str(PROJECT_ROOT.parent / plot_path)))
"""),
        markdown("## final check"),
        code(f"""
success_count = int(status['status'].eq('success').sum())
failed_count = int(status['status'].isin(['failed', 'qc_failed']).sum())
unfinished_count = int(status['status'].ne('success').sum())
print(f'{label} final status: success={{success_count}} of {{len(status)}}, unfinished={{unfinished_count}}, failed={{failed_count}}')
if unfinished_count and RUN_PREPROCESSING:
    raise RuntimeError(f'{label} has {{unfinished_count}} scans requiring checkpoint retry')
if unfinished_count:
    print('{label} report saved with unresolved scans requiring manual review')
else:
    print('{label} preprocessing and automated qc finished')
"""),
    ]
    nbformat.write(notebook, path)
    print(path)
