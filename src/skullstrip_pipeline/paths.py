from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT.parent
PROCESSED_ROOT = DATA_ROOT / "Processed"
PROCESSED_MRI_ROOT = PROCESSED_ROOT / "MRI"
MANIFEST_ROOT = DATA_ROOT / "Manifests"
SOURCE_MANIFEST = MANIFEST_ROOT / "all_datasets_detailed_manifest.csv"
AIBL_RAW_ROOT = DATA_ROOT / "AIBL"
AIBL_RAW3D_ROOT = AIBL_RAW_ROOT / "raw3DAll"
AIBL_RAW_MANIFEST_ROOT = AIBL_RAW_ROOT / "manifests"
AIBL_CONVERSION_MANIFEST = AIBL_RAW_MANIFEST_ROOT / "aibl_2d_to_3d_manifest.csv"
MNI_TEMPLATE = PROJECT_ROOT / "resources" / "templates" / "MNI152_T1_1mm_brain.nii.gz"


def dataset_output_root(dataset: str) -> Path:
    return PROCESSED_MRI_ROOT / dataset.upper()
