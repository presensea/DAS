"""
DistraSense file reader  (efficient rewrite, part of the das_audit package).

Reads DistraSense binary data files (`.ds`, `.sd`, `.dat`): a fixed 128-byte
header, a measurement payload, and a trailing 20-byte SHA-1 checksum.

Efficiency notes vs. the original implementation
-------------------------------------------------
* The payload is read with ``numpy.fromfile`` straight into a typed array,
  instead of ``file.read()`` → ``bytes[:-20]`` slice → ``np.frombuffer``.
  That removes two full-payload copies per file.
* ``read_multiple_files`` writes the per-file phase conversion directly into
  the pre-allocated output buffer with ``np.multiply(..., out=...)``,
  avoiding a separate temporary float array per file.

Public API (unchanged — drop-in compatible):
    calculate_checksum(data)                                    -> bytes
    read_distrasense_header(file_handle, minimal, verify_checksum) -> dict
    convert_phase_data(data)                                    -> ndarray
    read_distrasense_file(filepath, minimal, verify_checksum)   -> (ndarray, dict)
    read_multiple_files(folder, file_extension, max_files, channel_slice)
                                                                -> (ndarray, dict)
"""

import datetime
import hashlib
import os
import struct
import uuid
import warnings
from typing import Dict, Optional, Tuple

import numpy as np

HEADER_LEN = 128
CHECKSUM_LEN = 20
# Raw C-OTDR integer counts → optical phase in radians.
PHASE_SCALE = np.pi / 20589.0

# data_type code → on-disk NumPy dtype of the payload (big-endian).
_DTYPE_MAP = {
    2: ">i4",  # C-OTDR raw
    3: ">u8",  # B-OTDR
    5: ">i4",  # A-OTDR raw
    7: ">f8",  # C-OTDR processed
    9: ">f8",  # A-OTDR processed
}
_PHASE_TYPE = 2  # data_type that needs the radian conversion

# The 128-byte header is a fully contiguous, big-endian, unpadded record.
# Compile the layout once so each header is parsed in a single C-level
# struct.unpack call instead of ~25 separate ones.
_HEADER_STRUCT = struct.Struct(
    "!"     # big-endian, standard sizes, no alignment padding
    "4s"    # magic
    "H"     # header_version
    "H"     # data_format_version
    "I"     # sw_version
    "I"     # fw_version
    "I"     # total_header_len
    "I"     # chunk_number
    "Q"     # timestamp
    "16s"   # measurement_id
    "16s"   # device_id
    "I"     # pulse_width
    "H"     # rep_frequency
    "I"     # test_distance_a
    "H"     # measurement_time
    "I"     # test_distance_b
    "H"     # avg_times
    "H"     # start_freq
    "H"     # end_freq
    "H"     # step_size
    "I"     # test_distance_c
    "H"     # merging_points
    "H"     # differential_delay
    "I"     # data_type
    "I"     # num_frames
    "I"     # points_per_frame
    "20s"   # header_checksum
)
assert _HEADER_STRUCT.size == HEADER_LEN, "header layout must total 128 bytes"


def calculate_checksum(data: bytes) -> bytes:
    """SHA-1 digest (20 bytes) of ``data``."""
    return hashlib.sha1(data).digest()


# ── Header ──────────────────────────────────────────────────────────────

def _parse_header(header: bytes, minimal: bool = False,
                   verify_checksum: bool = False) -> Dict:
    """Parse a 128-byte DistraSense header block in a single unpack call."""
    if len(header) < HEADER_LEN:
        raise ValueError("File too short to contain a valid DistraSense header")

    (magic_b, header_version, data_format_version, sw, fw,
     total_header_len, chunk_number, timestamp, mid_b, did_b,
     pulse_width, rep_frequency, test_distance_a, measurement_time,
     test_distance_b, avg_times, start_freq, end_freq, step_size,
     test_distance_c, merging_points, diff_delay, data_type,
     num_frames, points_per_frame, header_checksum
     ) = _HEADER_STRUCT.unpack(header[:HEADER_LEN])

    magic = magic_b.decode("utf-8", errors="ignore")
    measurement_id = str(uuid.UUID(bytes=mid_b))

    if verify_checksum and calculate_checksum(header[0:108]) != header_checksum:
        warnings.warn(f"Header checksum verification failed for {measurement_id}")

    if minimal:
        return {
            "magic": magic,
            "total_header_len": total_header_len,
            "chunk_number": chunk_number,
            "timestamp": timestamp,
            "measurement_id": measurement_id,
            "data_type": data_type,
            "num_frames": num_frames,
            "points_per_frame": points_per_frame,
        }

    software_version = f"{(sw >> 24) & 0xFF}.{(sw >> 16) & 0xFF}.{(sw >> 8) & 0xFF}.{sw & 0xFF}"
    firmware_version = f"{(fw >> 24) & 0xFF}.{(fw >> 16) & 0xFF}.{(fw >> 8) & 0xFF}.{fw & 0xFF}"
    device_id = str(uuid.UUID(bytes=did_b))

    return {
        "magic": magic,
        "header_version": header_version,
        "data_format_version": data_format_version,
        "software_version": software_version,
        "firmware_version": firmware_version,
        "total_header_len": total_header_len,
        "chunk_number": chunk_number,
        "timestamp": timestamp,
        "timestamp_readable": datetime.datetime.fromtimestamp(
            timestamp / 1e9).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "measurement_id": measurement_id,
        "device_id": device_id,
        "pulse_width": pulse_width,
        "rep_frequency": rep_frequency,
        "test_distance_a": test_distance_a,
        "measurement_time": measurement_time,
        "test_distance_b": test_distance_b,
        "avg_times": avg_times,
        "start_freq": start_freq,
        "end_freq": end_freq,
        "step_size": step_size,
        "test_distance_c": test_distance_c,
        "merging_points": merging_points,
        "differential_delay": diff_delay,
        "data_type": data_type,
        "num_frames": num_frames,
        "points_per_frame": points_per_frame,
    }


def read_distrasense_header(file_handle, minimal: bool = False,
                            verify_checksum: bool = False) -> Dict:
    """Read and parse the header from an open file positioned at byte 0."""
    return _parse_header(file_handle.read(HEADER_LEN), minimal, verify_checksum)


# ── Payload ─────────────────────────────────────────────────────────────

def convert_phase_data(data: np.ndarray) -> np.ndarray:
    """Convert raw C-OTDR integer counts to optical phase in radians."""
    return data * PHASE_SCALE


def _read_raw(filepath: str, verify_checksum: bool = False) -> Tuple[np.ndarray, Dict]:
    """Read header + raw (un-converted) payload, reshaped to (frames, points).

    Returns the payload in its on-disk dtype — no phase conversion applied.
    """
    with open(filepath, "rb") as f:
        params = _parse_header(f.read(HEADER_LEN), minimal=False,
                               verify_checksum=verify_checksum)
        dtype = np.dtype(_DTYPE_MAP.get(params["data_type"], ">i4"))
        num_frames = params["num_frames"]
        ppf = params["points_per_frame"]
        expected = num_frames * ppf

        # Read the payload straight into a typed array (no intermediate copy).
        f.seek(HEADER_LEN)
        data = np.fromfile(f, dtype=dtype, count=expected)

        if verify_checksum:
            stored = f.read(CHECKSUM_LEN)
            if len(stored) == CHECKSUM_LEN:
                if calculate_checksum(data.tobytes()) != stored:
                    warnings.warn(
                        f"Data checksum verification failed for "
                        f"{params['measurement_id']}")

    if data.size != expected:
        warnings.warn(f"Data length {data.size} does not match expected {expected}")
        num_frames = data.size // ppf if ppf else 0
        params["actual_num_frames"] = num_frames
        data = data[:num_frames * ppf]

    if num_frames > 0 and ppf > 0:
        data = data.reshape(num_frames, ppf)
    return data, params


def read_distrasense_file(filepath: str, minimal: bool = False,
                          verify_checksum: bool = False) -> Tuple[np.ndarray, Dict]:
    """Read a single DistraSense file. Returns ``(data, parameters)``.

    For C-OTDR raw files (data_type 2) the integer payload is converted to
    phase in radians; other types are returned in their native dtype.
    ``minimal`` is accepted for API compatibility (the full header is always
    parsed — it is only 128 bytes).
    """
    data, params = _read_raw(filepath, verify_checksum=verify_checksum)
    if params["data_type"] == _PHASE_TYPE:
        data = data * PHASE_SCALE          # → float64 phase in radians
    return data, params


def read_multiple_files(folder_path: str, file_extension: str = ".ds",
                        max_files: int = 0,
                        channel_slice: Optional[slice] = None,
                        ) -> Tuple[np.ndarray, Dict]:
    """Read and stack every DistraSense file in a directory.

    Args:
        folder_path:    directory containing the files.
        file_extension: extension to match (``.ds``, ``.sd``, ``.dat``).
        max_files:      cap the number of files (0 = all).
        channel_slice:  optional column slice applied to every file.

    Returns ``(stacked_data, parameters_of_first_file)``. For C-OTDR raw data
    the phase conversion is written directly into the output buffer, so no
    per-file temporary float array is allocated.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Directory not found: {folder_path}")

    filenames = sorted(
        fn for fn in os.listdir(folder_path)
        if fn.endswith(file_extension) and not fn.startswith("._")
    )
    if not filenames:
        raise ValueError(f"No {file_extension} files found in {folder_path}")
    if max_files > 0:
        filenames = filenames[:max_files]
    n_files = len(filenames)

    # First file fixes geometry, dtype and the returned parameters.
    raw0, params = _read_raw(os.path.join(folder_path, filenames[0]))
    if channel_slice is not None:
        raw0 = raw0[:, channel_slice]
    frames_per_file, n_channels = raw0.shape
    is_phase = params["data_type"] == _PHASE_TYPE
    out_dtype = np.float64 if is_phase else raw0.dtype

    out = np.empty((frames_per_file * n_files, n_channels), dtype=out_dtype)
    if is_phase:
        np.multiply(raw0, PHASE_SCALE, out=out[:frames_per_file])
    else:
        out[:frames_per_file] = raw0
    del raw0

    n_loaded = 1
    for i, fn in enumerate(filenames[1:], start=1):
        try:
            raw, _ = _read_raw(os.path.join(folder_path, fn))
        except Exception as e:                       # noqa: BLE001
            warnings.warn(f"Failed to read {fn}: {e}")
            continue
        if channel_slice is not None:
            raw = raw[:, channel_slice]
        r0 = i * frames_per_file
        r1 = min(r0 + raw.shape[0], out.shape[0])
        if r1 <= r0:
            continue
        raw = raw[:r1 - r0]
        if is_phase:
            np.multiply(raw, PHASE_SCALE, out=out[r0:r1])
        else:
            out[r0:r1] = raw
        n_loaded += 1

    if n_loaded == 0:
        raise ValueError("No valid measurements could be read")

    return out[:n_loaded * frames_per_file], params


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        d, p = read_distrasense_file(sys.argv[1])
        print(f"File:  {sys.argv[1]}")
        print(f"Shape: {d.shape}  dtype={d.dtype}")
        print(f"Measurement ID: {p['measurement_id']}")
        print(f"Timestamp:      {p.get('timestamp_readable', p['timestamp'])}")
        print(f"data_type={p['data_type']}  frames={p['num_frames']}  "
              f"points/frame={p['points_per_frame']}")
