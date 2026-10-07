#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from skullstrip_pipeline.aibl_conversion import run_aibl_conversion
from skullstrip_pipeline.driver import public_status_summary, run_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Resumable AIBL conversion and SynthStrip preprocessing pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert-aibl")
    convert.add_argument("--max-files", type=int, default=5, help="number to process; use 0 for all pending files")
    convert.add_argument("--workers", type=int, default=2)
    convert.add_argument("--replace", action="store_true")
    convert.add_argument("--stop-on-error", action="store_true")

    strip = subparsers.add_parser("skullstrip")
    strip.add_argument("--dataset", required=True, choices=["ADNI", "AIBL", "OASIS2", "OASIS3"])
    strip.add_argument("--max-files", type=int, default=3, help="number to process; use 0 for all pending files")
    strip.add_argument("--replace", action="store_true")
    strip.add_argument("--no-qc-plots", action="store_true")
    strip.add_argument("--max-qc-plots", type=int, default=3)
    strip.add_argument("--gpu-id", type=int, default=2)

    args = parser.parse_args()
    max_files = None if args.max_files == 0 else args.max_files
    if args.command == "convert-aibl":
        result = run_aibl_conversion(max_files, args.replace, args.workers, args.stop_on_error)
        print(result.groupby("status").size().rename("count").to_string())
    else:
        result = run_dataset(
            args.dataset, max_files, args.replace, not args.no_qc_plots,
            args.max_qc_plots, gpu_physical_id=args.gpu_id,
        )
        print(public_status_summary(result).to_string(index=False))


if __name__ == "__main__":
    main()
