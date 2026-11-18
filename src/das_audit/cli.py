#!/usr/bin/env python3
"""
das-audit: DAS Data Quality Audit CLI

Usage:
  das-audit --dataset baltic_sea --files *.zarr --export-html
  das-audit --path /path/to/das/data --export-html
  das-audit --path /path/to/data.zarr --convert-zarr --export-html
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from das_audit.checks import (
    check_channel_rms,
    check_coupling_coherence,
    check_fk_aliasing,
    check_metadata,
    check_multiband_waterfall,
    check_saturation,
    check_spectral_baseline,
    check_time_continuity,
    check_wake_energy,
)
from das_audit.ingest import auto_load, convert_ds_to_zarr, discover_files
from das_audit.provenance import log_audit
from das_audit.report import generate_html_report


def run_audit(data: np.ndarray, header: dict, fs: float, timestamps=None,
              dx: float = 1.0, wake: bool = False):
    """Run all QA checks on a loaded dataset."""
    checks = []
    n_checks = 9 if wake else 8

    print(f"  [1/{n_checks}] Metadata validation...")
    checks.append(check_metadata(header))

    print(f"  [2/{n_checks}] Saturation check...")
    checks.append(check_saturation(data))

    print(f"  [3/{n_checks}] Per-channel RMS...")
    checks.append(check_channel_rms(data))

    print(f"  [4/{n_checks}] Spectral baseline...")
    checks.append(check_spectral_baseline(data, fs))

    print(f"  [5/{n_checks}] Time continuity...")
    if timestamps is not None and len(timestamps) > 1:
        checks.append(check_time_continuity(timestamps))
    else:
        # Synthesize timestamps from frame count and fs
        n_frames = data.shape[0]
        synth_ts = np.arange(n_frames) / fs
        checks.append(check_time_continuity(synth_ts, expected_dt=1.0 / fs))

    print(f"  [6/{n_checks}] Coupling coherence...")
    checks.append(check_coupling_coherence(data))

    print(f"  [7/{n_checks}] f-k aliasing check...")
    checks.append(check_fk_aliasing(data, fs, dx=dx))

    print(f"  [8/{n_checks}] Multi-band waterfall analysis...")
    checks.append(check_multiband_waterfall(data, fs, dx=dx))

    if wake:
        print(f"  [9/{n_checks}] Wake-line analysis (DAShip)...")
        checks.append(check_wake_energy(data, fs, dx=dx))

    return checks


def main():
    parser = argparse.ArgumentParser(
        prog="das-audit",
        description="DAS Data Quality Audit — ingest, grade, and report.",
    )
    parser.add_argument("--path", "-p", required=True,
                        help="Path to DAS data file or directory (.ds, .sd, .zarr, .h5)")
    parser.add_argument("--dataset", "-d", default=None,
                        help="Dataset name for provenance logging")
    parser.add_argument("--files", "-f", default=None,
                        help="Glob pattern for files within --path (e.g., '*.zarr')")
    parser.add_argument("--export-html", action="store_true",
                        help="Generate interactive HTML report")
    parser.add_argument("--output", "-o", default=None,
                        help="Output path for HTML report (default: das_audit_report.html)")
    parser.add_argument("--convert-zarr", action="store_true",
                        help="Convert raw .ds/.sd files to zarr before auditing")
    parser.add_argument("--zarr-output", default=None,
                        help="Output path for zarr conversion")
    parser.add_argument("--dx", type=float, default=1.0,
                        help="Channel spacing in meters (default: 1.0)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max number of .ds files to load (0 = all, default: 0)")
    parser.add_argument("--channels", type=str, default=None,
                        help="Channel range to select, e.g. '5000:6300' for submarine cable")
    parser.add_argument("--wake", action="store_true",
                        help="Run DAShip wake-line analysis (use with submarine cable channels)")
    parser.add_argument("--provenance-db", default="das_audit_provenance.parquet",
                        help="Path to provenance Parquet database")
    parser.add_argument("--no-provenance", action="store_true",
                        help="Skip provenance logging")

    args = parser.parse_args()

    path = args.path
    dataset_name = args.dataset or os.path.basename(path)

    print(f"das-audit v0.1.0")
    print(f"Dataset: {dataset_name}")
    print(f"Path: {path}")

    # Optional conversion
    if args.convert_zarr:
        zarr_out = args.zarr_output or (path.rstrip("/") + ".zarr")
        print(f"\nConverting to zarr: {zarr_out}")
        convert_ds_to_zarr(path, zarr_out)
        path = zarr_out

    # Load data
    print(f"\nLoading data...")
    t0 = time.time()

    if args.files:
        # Glob within directory
        pattern = os.path.join(path, args.files)
        matched = sorted(glob.glob(pattern))
        if not matched:
            print(f"No files matching {pattern}")
            sys.exit(1)
        # Load first match
        path = matched[0]
        print(f"  Using: {path}")

    # Parse channel range
    channel_slice = None
    if args.channels:
        parts = args.channels.split(":")
        ch_start = int(parts[0]) if parts[0] else None
        ch_end = int(parts[1]) if len(parts) > 1 and parts[1] else None
        channel_slice = slice(ch_start, ch_end)
        print(f"  Channel selection: {args.channels}")

    data, header, fs = auto_load(path, max_files=args.max_files,
                                  channel_slice=channel_slice)
    load_time = time.time() - t0

    print(f"  Shape: {data.shape}")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Duration: {data.shape[0] / fs:.1f} s")
    print(f"  Channels: {data.shape[1]}")
    print(f"  Loaded in {load_time:.1f}s")

    # Run audit
    print(f"\nRunning QA checks...")
    checks = run_audit(data, header, fs, dx=args.dx, wake=args.wake)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"AUDIT RESULTS — {dataset_name}")
    print(f"{'=' * 60}")
    for chk in checks:
        s = chk.get("status", "pass" if chk["passed"] else "fail")
        label = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}.get(s, s.upper())
        print(f"  [{label}] {chk['name']}: {chk['summary']}")

    n_fail = sum(1 for c in checks if c.get("status", "pass" if c["passed"] else "fail") == "fail")
    n_warn = sum(1 for c in checks if c.get("status") == "warn")
    n_pass = len(checks) - n_fail - n_warn
    print(f"\nOverall: {n_pass} pass, {n_warn} warn, {n_fail} fail")

    # Provenance
    if not args.no_provenance:
        db_path = log_audit(checks, header, dataset_name, args.provenance_db)
        print(f"Provenance logged to: {db_path}")

    # HTML report
    if args.export_html:
        output = args.output or f"das_audit_{dataset_name}.html"
        meta_display = {
            "Dataset": dataset_name,
            "Path": args.path,
            "Shape": f"{data.shape[0]} frames x {data.shape[1]} channels",
            "Sample Rate": f"{fs} Hz",
            "Duration": f"{data.shape[0] / fs:.1f} s",
            "Data Type": header.get("data_type", "unknown"),
            "Measurement ID": header.get("measurement_id", "N/A"),
            "Timestamp": header.get("timestamp_readable", "N/A"),
            "Pulse Width": f"{header.get('pulse_width', 'N/A')} ns",
            "Test Distance": f"{header.get('test_distance_c', header.get('test_distance_b', 'N/A'))} m",
        }
        generate_html_report(checks, meta_display, output)
        print(f"HTML report: {output}")

    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
