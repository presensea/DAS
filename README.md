# das-audit — DAS Data Quality Audit Tool

A lightweight, low-RAM command-line tool for automated quality assessment of Distributed Acoustic Sensing (DAS) datasets. It ingests raw DistraSense `.ds`/`.sd`/`.dat` files as well as `.zarr` and `.h5`/`.hdf5` archives, runs sequential signal-quality checks, logs provenance, and generates a self-contained interactive HTML dashboard using Plotly.js.

Developed within the **PRESENSEA** project (*MOB3JD1212: Pressure Sensor Based Sea Monitoring*, [presensea.github.io](https://presensea.github.io)) at the Centre for Biorobotics, Tallinn University of Technology (TalTech).

## Installation

```bash
git clone https://github.com/presensea/DAS.git
cd DAS

# Core dependencies (numpy, scipy)
pip install -e .

# Optional extras for Zarr, HDF5, and Parquet provenance logging
pip install -e ".[all]"
```

You can also run directly from the repository root without installing:

```bash
./das-audit --help
```

## Quick Start

```bash
# Audit first 5 minutes of a recording (300 files at 1 file/s)
das-audit \
  --path /path/to/das/data \
  --max-files 300 \
  --dx 5.0 \
  --export-html

# Audit a specific submarine channel slice (5000–6300) with DAShip wake analysis
das-audit \
  --path /path/to/das/data \
  --channels 5000:6300 \
  --max-files 300 \
  --dx 5.0 \
  --wake \
  --export-html

# Convert raw .ds files to Zarr first, then audit
das-audit \
  --path /path/to/ds_folder \
  --convert-zarr \
  --export-html
```

## Architecture

```text
Raw .ds/.sd/.dat / .zarr / .h5
    │
    ├── distrasense_reader.py   Binary parser (128-byte header + int32/float64 payload + SHA-1 checksum)
    v
ingest.py: auto_load()          Format detection, channel slicing at read time, max_files cap
    │
    v
checks.py: run_audit()          8 standard QA checks (+ optional DAShip wake-line analysis)
    │
    v
report.py: generate_html_report()   Self-contained interactive HTML report (Plotly.js)
    │
    v
provenance.py: log_audit()      Append run metrics to Parquet database (with JSONL fallback)
```

## QA Checks

| # | Check | What it does | Pass / Status condition |
|---|-------|-------------|-------------------------|
| 1 | **Metadata Validation** | Verifies required header fields (`pulse_width`, `rep_frequency`, `points_per_frame`, `num_frames`) and gauge length | All required fields present and non-zero |
| 2 | **Saturation Check** | Flags NaN and clipped samples | `< 0.1%` NaN and `< 0.1%` clipping |
| 3 | **Per-Channel RMS** | Computes per-channel standard deviation | No dead (`RMS = 0`) or saturated (`> 10x` median) channels |
| 4 | **Spectral Baseline** | Welch PSD on sampled channels and 50/60 Hz mains peaks | No electrical noise peaks `> 5x` local median |
| 5 | **Time Continuity** | Analyses timestamp diffs for dropped-packet gaps | No gaps `> 1.5x` expected `dt` |
| 6 | **Coupling Coherence** | Normalized cross-correlation (NCC) between adjacent channels | `< 10%` channels with `NCC < 0.3` |
| 7 | **f-k Aliasing** | 2D FFT edge-vs-center energy ratio | Edge/center energy ratio `< 0.5` |
| 8 | **Multi-Band Waterfall** | Bandpass-filtered, z-scored, common-mode-removed waterfalls | Informational visual inspection |
| 9 | **Wake-Line Analysis (`--wake`)** | DAShip-inspired downsample, z-score, bandpass & f-k velocity filter | Informational spatial contrast & waterfalls |

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--path`, `-p` | *(required)* | Path to DAS data file or directory (`.ds`, `.sd`, `.zarr`, `.h5`) |
| `--dataset`, `-d` | `dirname` | Dataset name for provenance logging |
| `--files`, `-f` | `None` | Glob pattern for files within `--path` (e.g., `*.zarr`) |
| `--max-files` | `0` (all) | Limit number of `.ds`/`.sd` files loaded (RAM control) |
| `--channels` | all | Channel slice, e.g. `5000:6300` |
| `--dx` | `1.0` | Channel spacing in meters |
| `--wake` | off | Run DAShip wake-line analysis (Check 9) |
| `--export-html` | off | Generate interactive HTML report |
| `--output`, `-o` | auto | Output path for HTML report |
| `--convert-zarr` | off | Convert raw `.ds`/`.sd` folder to Zarr before auditing |
| `--zarr-output` | auto | Output path for Zarr conversion |
| `--provenance-db` | `das_audit_provenance.parquet` | Path to provenance Parquet database |
| `--no-provenance` | off | Skip provenance logging |

## Memory Management

* **`--max-files N`**: Caps the number of raw files loaded from a folder.
* **`--channels START:END`**: Applies channel slicing during binary read rather than after loading the full array.
* Pre-allocated NumPy output arrays avoid doubling peak memory during phase conversion.

## File Format Reference

DistraSense `.ds` binary files:
* 128-byte big-endian header (`DSTR` magic, UUIDs, measurement parameters) + 20-byte SHA-1 checksum
* Payload (`>i4` for raw C-OTDR, converted to optical phase via `data * pi / 20589`) + trailing 20-byte SHA-1 checksum
* Shape per file: `num_frames x points_per_frame`

## Dependencies

* **Required:** `numpy`, `scipy`
* **Optional:** `zarr`, `h5py`, `pandas`, `pyarrow`

## Funding

This project has received funding from the Estonian Research Council (ETAG) under the Mobilitas Pluss / Mobilitas 3.0 programme (Grant no. `MOB3JD1212`, *Pressure Sensor Based Sea Monitoring*) and is co-funded by the European Union.
