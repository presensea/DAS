"""
Data ingestion: read raw .ds/.sd files, .zarr, and .h5 into a common format.
Optionally convert raw DistraSense files to zarr.
"""

import os
import glob
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# Reader is now part of the das_audit package.
from .distrasense_reader import read_distrasense_file, read_multiple_files


def discover_files(path: str, extensions: tuple = (".ds", ".sd", ".dat", ".zarr", ".h5")) -> dict:
    """Discover DAS data files in a directory tree."""
    result = {ext: [] for ext in extensions}
    p = Path(path)

    if p.is_file():
        ext = p.suffix.lower()
        if ext in extensions:
            result[ext] = [str(p)]
        return result

    for ext in extensions:
        pattern = f"**/*{ext}"
        result[ext] = sorted(str(f) for f in p.glob(pattern))

    return result


def load_distrasense(filepath: str, verify_checksum: bool = False):
    """Load a single .ds or .sd file. Returns (data, header)."""
    return read_distrasense_file(filepath, verify_checksum=verify_checksum)


def load_distrasense_folder(folder: str, extension: str = ".ds",
                            verify_checksum: bool = False, max_files: int = 0,
                            channel_slice: slice = None):
    """Load files from a folder and stack. Returns (data, header).

    Args:
        max_files: Limit number of files loaded (0 = all).
        channel_slice: Optional slice to select channel subset per file.
    """
    return read_multiple_files(folder, file_extension=extension,
                               max_files=max_files, channel_slice=channel_slice)


def load_zarr(filepath: str):
    """Load a zarr array. Returns (data, metadata_dict)."""
    import zarr
    z = zarr.open(filepath, mode="r")

    if hasattr(z, "shape"):
        data = z[:]
        attrs = dict(z.attrs) if hasattr(z, "attrs") else {}
    else:
        # zarr group — find the main array
        keys = list(z.keys())
        if "data" in keys:
            data = z["data"][:]
        elif len(keys) == 1:
            data = z[keys[0]][:]
        else:
            data = z[keys[0]][:]
            warnings.warn(f"Multiple arrays in zarr group, using '{keys[0]}'. Available: {keys}")
        attrs = dict(z.attrs) if hasattr(z, "attrs") else {}

    return data, attrs


def load_h5(filepath: str, dataset_key: str = None):
    """Load an HDF5 file. Returns (data, attrs_dict)."""
    import h5py
    with h5py.File(filepath, "r") as f:
        if dataset_key:
            data = f[dataset_key][:]
            attrs = dict(f[dataset_key].attrs)
        else:
            keys = list(f.keys())
            if "data" in keys:
                key = "data"
            elif len(keys) == 1:
                key = keys[0]
            else:
                key = keys[0]
                warnings.warn(f"Multiple datasets in H5, using '{key}'. Available: {keys}")
            data = f[key][:]
            attrs = dict(f[key].attrs)

    return data, attrs


def convert_ds_to_zarr(input_dir: str, output_path: str, extension: str = ".ds",
                       chunk_seconds: int = 60) -> str:
    """Convert a folder of .ds files into a single zarr array."""
    import zarr

    data, header = read_multiple_files(input_dir, file_extension=extension)

    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store, overwrite=True)

    # Choose chunk shape: chunk_seconds worth of frames
    fs = header.get("rep_frequency", 1)
    chunk_frames = max(1, chunk_seconds * fs)
    chunks = (min(chunk_frames, data.shape[0]), data.shape[1])

    root.create_dataset("data", data=data, chunks=chunks, dtype=data.dtype)

    # Store header as attrs
    for k, v in header.items():
        try:
            root.attrs[k] = v
        except (TypeError, ValueError):
            root.attrs[k] = str(v)

    print(f"Converted {input_dir} -> {output_path}  shape={data.shape}")
    return output_path


def auto_load(filepath: str, max_files: int = 0, channel_slice: slice = None):
    """Auto-detect format and load. Returns (data_2d, metadata_dict, fs).

    Args:
        max_files: For .ds/.sd folders, limit number of files (0 = all).
        channel_slice: Optional slice to select a subset of channels.
    """
    p = Path(filepath)

    sliced = False  # track whether channel_slice was applied during load
    if p.is_dir():
        # Check for zarr or ds files
        if any(p.glob("*.zarr")) or (p / ".zarray").exists():
            data, meta = load_zarr(str(p))
        elif any(p.glob("*.ds")):
            data, meta = load_distrasense_folder(str(p), ".ds",
                                                  max_files=max_files,
                                                  channel_slice=channel_slice)
            sliced = True
        elif any(p.glob("*.sd")):
            data, meta = load_distrasense_folder(str(p), ".sd",
                                                  max_files=max_files,
                                                  channel_slice=channel_slice)
            sliced = True
        else:
            raise ValueError(f"No recognized DAS files in {p}")
    else:
        ext = p.suffix.lower()
        if ext in (".ds", ".sd", ".dat"):
            data, meta = load_distrasense(str(p))
        elif ext == ".zarr":
            data, meta = load_zarr(str(p))
        elif ext in (".h5", ".hdf5"):
            data, meta = load_h5(str(p))
        else:
            raise ValueError(f"Unsupported format: {ext}")

    # Ensure 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Apply channel slice for formats that don't handle it during load
    if channel_slice is not None and not sliced and data.ndim == 2:
        data = data[:, channel_slice]

    # Extract sample rate
    fs = meta.get("rep_frequency", meta.get("fs", meta.get("sample_rate", None)))
    if fs is None:
        warnings.warn("Could not determine sample rate from metadata. Defaulting to 1000 Hz.")
        fs = 1000

    return data, meta, float(fs)
