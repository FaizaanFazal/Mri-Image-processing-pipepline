#!/usr/bin/env python3
"""Remove participant-level images and local paths from public notebooks."""

from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
DRIVER_NOTEBOOKS = tuple(sorted((NOTEBOOK_ROOT / "drivers").glob("0[4-7]*_driver.ipynb")))
FORBIDDEN_TEXT = (
    re.compile(r"/(?:media|home)/[^/'\"\s]+/"),
    re.compile(r"\b\d{3}_S_\d{4,5}\b"),
    re.compile(r"\bOAS3\d{4}\b|\bOAS2_\d{4}\b"),
)


def output_has_image(output: dict[str, object]) -> bool:
    data = output.get("data", {})
    return isinstance(data, dict) and any(
        mime in data for mime in ("image/png", "image/jpeg", "image/svg+xml")
    )


def sanitize_driver(path: Path) -> tuple[int, int]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    removed_images = 0
    source_edits = 0
    safe_aggregate_image_kept = False

    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        private_display = (
            "for plot_path in visual_selection['plot_path']:\n"
            "    display(Image(filename=str(PROJECT_ROOT.parent / plot_path)))\n"
        )
        if private_display in source:
            source = source.replace(private_display, "")
            cell["source"] = source.splitlines(keepends=True)
            source_edits += 1

        retained = []
        for output in cell.get("outputs", []):
            if not output_has_image(output):
                retained.append(output)
                continue
            if not safe_aggregate_image_kept:
                # The first displayed image is qc_summary.png, which contains
                # only cohort-level distributions and no participant anatomy.
                retained.append(output)
                safe_aggregate_image_kept = True
            else:
                removed_images += 1
        if "outputs" in cell:
            cell["outputs"] = retained

    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return removed_images, source_edits


def make_clinical_copy_portable() -> int:
    path = NOTEBOOK_ROOT / "clinical_copy" / "clinical_data_copy_and_manifests.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    machine_path = re.compile(
        r"^PROJECT_ROOT = Path\(['\"]/(?:media|home)/.*?['\"]\)\n",
        flags=re.MULTILINE,
    )
    new = (
        "project_candidates = [Path.cwd(), *Path.cwd().parents, Path.cwd() / 'SkullStrippingProject']\n"
        "CODE_ROOT = next(path for path in project_candidates if (path / 'src' / 'skullstrip_pipeline').exists())\n"
        "PROJECT_ROOT = CODE_ROOT.parent\n"
    )
    edits = 0
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if machine_path.search(source):
            cell["source"] = machine_path.sub(new, source).splitlines(keepends=True)
            edits += 1
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return edits


def validate() -> None:
    failures: list[str] = []
    for path in sorted(NOTEBOOK_ROOT.rglob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        serialized = json.dumps(notebook, ensure_ascii=False)
        for pattern in FORBIDDEN_TEXT:
            if pattern.search(serialized):
                failures.append(f"{path.relative_to(PROJECT_ROOT)} matches {pattern.pattern}")

        image_outputs = sum(
            output_has_image(output)
            for cell in notebook.get("cells", [])
            for output in cell.get("outputs", [])
        )
        if path in DRIVER_NOTEBOOKS and image_outputs > 1:
            failures.append(
                f"{path.relative_to(PROJECT_ROOT)} has {image_outputs} embedded images; expected at most one aggregate dashboard"
            )

    if failures:
        raise RuntimeError("public notebook validation failed:\n" + "\n".join(failures))


def main() -> None:
    removed = 0
    edits = 0
    for path in DRIVER_NOTEBOOKS:
        removed_here, edits_here = sanitize_driver(path)
        removed += removed_here
        edits += edits_here
    portable_edits = make_clinical_copy_portable()
    validate()
    print(f"sanitized {len(DRIVER_NOTEBOOKS)} driver notebooks")
    print(f"removed participant-level image outputs: {removed}")
    print(f"removed participant-image display loops: {edits}")
    print(f"replaced machine-specific paths: {portable_edits}")
    print("public notebook validation: passed")


if __name__ == "__main__":
    main()
