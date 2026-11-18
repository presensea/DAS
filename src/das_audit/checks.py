"""
QA check functions for DAS data audit.

Each check returns a dict with:
  - name: str
  - passed: bool
  - status: str ("pass", "warn", "fail")
  - summary: str
  - details: dict (check-specific data for plotting / HTML)
"""

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq


# ── Helpers ──────────────────────────────────────────────────────────────

def _bandpass(data, lo_hz, hi_hz, fs, order=4):
    """Zero-phase Butterworth bandpass. Returns filtered copy."""
    nyq = 0.5 * fs
    lo = max(lo_hz / nyq, 1e-6)
    hi = min(hi_hz / nyq, 0.9999)
    if lo >= hi:
        return data.copy()
    sos = butter(order, [lo, hi], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=0)


def _ncc_pair(a, b):
    """Normalized cross-correlation between two 1D arrays."""
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask] - np.mean(a[mask]), b[mask] - np.mean(b[mask])
    if len(a) < 10:
        return 0.0
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(np.sum(a * b) / denom) if denom > 0 else 0.0


# ── Check 1: Metadata ───────────────────────────────────────────────────

def check_metadata(header: dict) -> dict:
    """Check that essential metadata fields exist and are plausible."""
    required = ["pulse_width", "rep_frequency", "points_per_frame", "num_frames"]
    missing = [k for k in required if k not in header or header[k] == 0]

    gauge_length = header.get("test_distance_a", 0)
    output_rate = header.get("rep_frequency", 0)

    warnings = []
    if gauge_length == 0:
        warnings.append("gauge_length (test_distance_a) is 0 or missing")
    if output_rate == 0:
        warnings.append("rep_frequency is 0 or missing")

    return {
        "name": "Metadata Validation",
        "passed": len(missing) == 0 and len(warnings) == 0,
        "summary": f"{len(missing)} missing fields, {len(warnings)} warnings",
        "details": {
            "missing_fields": missing,
            "warnings": warnings,
            "gauge_length": gauge_length,
            "pitch_channels": header.get("points_per_frame", 0),
            "output_rate_hz": output_rate,
        },
    }


# ── Check 2: Saturation ─────────────────────────────────────────────────

def check_saturation(data: np.ndarray, clip_threshold: float = 0.001) -> dict:
    """Flag NaNs or clipping > threshold fraction."""
    n_total = data.size
    n_nan = int(np.isnan(data).sum())
    nan_frac = n_nan / n_total if n_total > 0 else 0.0

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        n_clip = int(np.sum((data == info.min) | (data == info.max)))
    else:
        if n_total > 0 and not np.all(np.isnan(data)):
            finite = data[np.isfinite(data)]
            if len(finite) > 100:
                lo, hi = np.percentile(finite, [0.01, 99.99])
                n_clip = int(np.sum((data <= lo) | (data >= hi)))
            else:
                n_clip = 0
        else:
            n_clip = 0

    clip_frac = n_clip / n_total if n_total > 0 else 0.0
    passed = nan_frac < clip_threshold and clip_frac < clip_threshold

    return {
        "name": "Saturation Check",
        "passed": passed,
        "summary": f"NaN: {nan_frac:.4%}, Clipping: {clip_frac:.4%}",
        "details": {
            "n_nan": n_nan, "nan_fraction": nan_frac,
            "n_clip": n_clip, "clip_fraction": clip_frac,
            "threshold": clip_threshold,
        },
    }


# ── Check 3: Per-channel RMS ────────────────────────────────────────────

def check_channel_rms(data: np.ndarray) -> dict:
    """Per-channel RMS (std dev). Dead channels = 0, saturated = spikes."""
    rms = np.nanstd(data, axis=0)
    median_rms = float(np.nanmedian(rms))
    dead_mask = rms == 0
    n_dead = int(dead_mask.sum())

    spike_mask = (rms > 10 * median_rms) if median_rms > 0 else np.zeros_like(rms, dtype=bool)
    n_spike = int(spike_mask.sum())

    n_ch = len(rms)
    dead_indices = set(np.where(dead_mask)[0].tolist())
    spike_indices = set(np.where(spike_mask)[0].tolist())
    bad_indices = dead_indices | spike_indices
    edge_channels = {0, n_ch - 1}

    edge_only = len(bad_indices) > 0 and bad_indices <= edge_channels
    if len(bad_indices) == 0:
        status = "pass"
    elif edge_only:
        status = "warn"
    else:
        status = "fail"

    return {
        "name": "Per-Channel RMS",
        "passed": status != "fail",
        "status": status,
        "summary": f"{n_dead} dead, {n_spike} saturated of {n_ch} channels"
                   + (" (edge only)" if edge_only else ""),
        "details": {
            "rms": rms.tolist(),
            "median_rms": median_rms,
            "n_dead": n_dead, "n_spike": n_spike,
            "dead_channels": sorted(dead_indices),
            "spike_channels": sorted(spike_indices),
            "edge_only": edge_only,
        },
    }


# ── Check 4: Spectral baseline (dense 50/60 Hz scan) ────────────────────

def check_spectral_baseline(data: np.ndarray, fs: float,
                            max_channels_scan: int = 200) -> dict:
    """Welch PSD — scan many channels for 50/60 Hz electrical noise.

    Computes PSD on up to max_channels_scan evenly-spaced channels and
    returns a per-channel 50 Hz power map for plotting.
    """
    n_ch = data.shape[1] if data.ndim == 2 else 1
    n_scan = min(max_channels_scan, n_ch)
    scan_idx = np.linspace(0, n_ch - 1, n_scan, dtype=int)

    nperseg = min(2048, data.shape[0])

    freqs = None
    # Per-channel 50 Hz and 60 Hz power ratios
    power_50 = np.zeros(n_scan)
    power_60 = np.zeros(n_scan)
    noise_peaks = []

    # Keep PSD curves for a sparser set (for plotting)
    plot_step = max(1, n_scan // 12)
    psd_plot_indices = list(range(0, n_scan, plot_step))
    psd_matrix = []
    psd_channels = []

    for i, ch in enumerate(scan_idx):
        col = data[:, ch] if data.ndim == 2 else data
        col = col[np.isfinite(col)]
        if len(col) < nperseg:
            continue
        f, pxx = welch(col, fs=fs, nperseg=nperseg)
        if freqs is None:
            freqs = f

        # Keep full PSD for plot channels
        if i in psd_plot_indices:
            psd_matrix.append(pxx.tolist())
            psd_channels.append(int(ch))

        # Check 50 Hz
        for target, arr in [(50.0, power_50), (60.0, power_60)]:
            if target < fs / 2:
                idx_t = np.argmin(np.abs(f - target))
                lo = max(0, idx_t - 10)
                hi = min(len(pxx), idx_t + 10)
                local_median = np.median(pxx[lo:hi])
                ratio = float(pxx[idx_t] / local_median) if local_median > 0 else 0.0
                arr[i] = ratio
                if ratio > 5:
                    noise_peaks.append({
                        "channel": int(ch), "freq_hz": target,
                        "power_ratio": ratio,
                    })

    has_electrical = len(noise_peaks) > 0
    n_ch_50 = int(np.sum(power_50 > 5))
    n_ch_60 = int(np.sum(power_60 > 5))

    return {
        "name": "Spectral Baseline",
        "passed": not has_electrical,
        "summary": (f"50 Hz peaks on {n_ch_50}/{n_scan} ch, "
                    f"60 Hz peaks on {n_ch_60}/{n_scan} ch"
                    if has_electrical else f"Clean (scanned {n_scan} channels)"),
        "details": {
            "freqs": freqs.tolist() if freqs is not None else [],
            "psd_matrix": psd_matrix,
            "psd_channels": psd_channels,
            "scan_channels": scan_idx.tolist(),
            "power_50hz": power_50.tolist(),
            "power_60hz": power_60.tolist(),
            "noise_peaks": noise_peaks,
            "n_channels_with_50hz": n_ch_50,
            "n_channels_with_60hz": n_ch_60,
            "fs": fs,
        },
    }


# ── Check 5: Time continuity ────────────────────────────────────────────

def check_time_continuity(timestamps: np.ndarray, expected_dt: float = None) -> dict:
    """Check timestamp diffs for dropped-packet gaps."""
    if len(timestamps) < 2:
        return {
            "name": "Time Continuity",
            "passed": False,
            "summary": "Insufficient timestamps",
            "details": {"n_timestamps": len(timestamps)},
        }

    diffs = np.diff(timestamps)
    if expected_dt is None:
        expected_dt = float(np.median(diffs))

    gap_mask = diffs > 1.5 * expected_dt
    n_gaps = int(gap_mask.sum())
    gap_indices = np.where(gap_mask)[0].tolist()
    gap_sizes = diffs[gap_mask].tolist()
    total_missing = float(np.sum(diffs[gap_mask] - expected_dt)) if n_gaps > 0 else 0.0

    return {
        "name": "Time Continuity",
        "passed": n_gaps == 0,
        "summary": f"{n_gaps} gaps, {total_missing:.2f}s missing",
        "details": {
            "expected_dt": expected_dt,
            "n_gaps": n_gaps,
            "gap_indices": gap_indices[:50],
            "gap_sizes_s": gap_sizes[:50],
            "total_missing_s": total_missing,
            "dt_std": float(np.std(diffs)),
        },
    }


# ── Check 6: Coupling coherence with cable segmentation ─────────────────

def check_coupling_coherence(data: np.ndarray, max_channels: int = 500,
                             rolling_window: int = 21) -> dict:
    """NCC of channel k with k+1, with rolling-median cable segmentation.

    Classifies cable segments using a rolling median of NCC:
      - buried/ground:  rolling NCC ≥ 0.6
      - conduit/loose:  rolling NCC 0.3–0.6
      - aerial/submarine: rolling NCC < 0.3
    """
    n_ch = data.shape[1] if data.ndim == 2 else 1
    step = max(1, n_ch // max_channels)
    channels = list(range(0, n_ch - 1, step))

    ncc_values = np.array([_ncc_pair(data[:, k], data[:, k + 1]) for k in channels])

    # Rolling median for smooth segmentation
    pad = rolling_window // 2
    padded = np.pad(ncc_values, pad, mode="edge")
    rolling_ncc = np.array([
        np.median(padded[i:i + rolling_window])
        for i in range(len(ncc_values))
    ])

    # Classify each channel
    seg_labels = np.where(
        rolling_ncc >= 0.6, 2,       # buried/ground
        np.where(rolling_ncc >= 0.3, 1, 0)  # conduit / aerial
    )
    LABEL_NAMES = {0: "aerial/submarine", 1: "conduit/loosely-coupled", 2: "buried/ground-coupled"}

    # Find contiguous segment boundaries
    segments = []
    seg_start = 0
    for i in range(1, len(seg_labels)):
        if seg_labels[i] != seg_labels[seg_start]:
            segments.append({
                "ch_start": channels[seg_start],
                "ch_end": channels[i - 1],
                "n_channels": i - seg_start,
                "type": LABEL_NAMES[int(seg_labels[seg_start])],
                "median_ncc": float(np.median(ncc_values[seg_start:i])),
            })
            seg_start = i
    segments.append({
        "ch_start": channels[seg_start],
        "ch_end": channels[-1],
        "n_channels": len(seg_labels) - seg_start,
        "type": LABEL_NAMES[int(seg_labels[seg_start])],
        "median_ncc": float(np.median(ncc_values[seg_start:])),
    })

    low_coupling = ncc_values < 0.3
    n_low = int(low_coupling.sum())
    low_frac = n_low / len(ncc_values) if len(ncc_values) > 0 else 0.0
    median_ncc = float(np.nanmedian(ncc_values))
    overall_type = LABEL_NAMES[int(np.median(seg_labels))]

    status = "pass" if low_frac < 0.1 else "warn"

    seg_summary = ", ".join(
        f"{s['type']} ch {s['ch_start']}–{s['ch_end']}" for s in segments
    )

    return {
        "name": "Coupling Coherence",
        "passed": status != "fail",
        "status": status,
        "summary": (f"median NCC {median_ncc:.2f} → {overall_type} | "
                    f"{len(segments)} segment(s)"),
        "details": {
            "ncc": ncc_values.tolist(),
            "rolling_ncc": rolling_ncc.tolist(),
            "channels": channels,
            "segments": segments,
            "seg_labels": seg_labels.tolist(),
            "n_low_coupling": n_low,
            "low_fraction": low_frac,
            "median_ncc": median_ncc,
            "overall_type": overall_type,
        },
    }


# ── Check 7: f-k aliasing ───────────────────────────────────────────────

def check_fk_aliasing(data: np.ndarray, fs: float, dx: float = 1.0,
                      max_samples: int = 2048) -> dict:
    """2D FFT to check whether spatial sampling resolves the highest frequencies."""
    n_t, n_ch = data.shape if data.ndim == 2 else (len(data), 1)

    t_slice = slice(0, min(n_t, max_samples))
    ch_slice = slice(0, min(n_ch, max_samples))
    block = np.nan_to_num(data[t_slice, ch_slice], nan=0.0)

    fk = np.abs(np.fft.fftshift(np.fft.fft2(block)))
    k_nyquist = 0.5 / dx
    f_nyquist = fs / 2

    k_edge = int(0.9 * block.shape[1] // 2)
    fk_norm = fk / (fk.max() + 1e-30)
    edge_energy = float(np.mean(fk_norm[:, -k_edge:]) + np.mean(fk_norm[:, :k_edge]))
    center_energy = float(np.mean(fk_norm[:, block.shape[1] // 4: 3 * block.shape[1] // 4]))
    aliasing_ratio = edge_energy / (center_energy + 1e-30)
    possibly_aliased = aliasing_ratio > 0.5

    return {
        "name": "f-k Aliasing",
        "passed": not possibly_aliased,
        "summary": f"Edge/center energy ratio: {aliasing_ratio:.2f} ({'ALIASED' if possibly_aliased else 'OK'})",
        "details": {
            "k_nyquist": k_nyquist, "f_nyquist": f_nyquist,
            "aliasing_ratio": aliasing_ratio,
            "fk_shape": list(fk.shape), "dx_m": dx, "fs_hz": fs,
        },
    }


# ── Check 8: Multi-band NCC analysis (time-resolved waterfalls) ────────

def check_multiband_waterfall(data: np.ndarray, fs: float, dx: float = 10.0) -> dict:
    """Bandpass-filtered waterfall plots per frequency band.

    For each band: bandpass → per-channel z-score → remove spatial common-mode
    (median across channels per time step). Stores downsampled waterfalls for
    the HTML report.

    Bands:
      - 0.1–1 Hz   (swell / wake / infragravity)
      - 1–10 Hz    (Scholte / engine)
      - 10–50 Hz   (propeller / machinery)
      - 50–200 Hz  (broadband ship noise)
      - 200–400 Hz (high-frequency)
    """
    n_t, n_ch = data.shape if data.ndim == 2 else (len(data), 1)

    f_nyq = fs / 2
    all_bands = [
        (0.1, 1.0, "0.1-1 Hz (swell/wake)"),
        (1.0, 10.0, "1-10 Hz (Scholte)"),
        (10.0, 50.0, "10-50 Hz (propeller)"),
        (50.0, 200.0, "50-200 Hz (broadband)"),
        (200.0, min(400.0, f_nyq * 0.95), "200-400 Hz (high-freq)"),
    ]
    bands = [(lo, hi, name) for lo, hi, name in all_bands if lo < f_nyq * 0.95]

    # Target waterfall size for HTML: max 800 time x 500 channels
    max_t, max_ch = 800, 500

    band_results = []
    for lo_hz, hi_hz, band_name in bands:
        print(f"    Band {band_name}...")

        # Downsample before filtering for low-freq bands
        if hi_hz <= 10.0:
            ds_f = max(1, int(fs / (hi_hz * 4)))
        else:
            ds_f = max(1, int(fs / (hi_hz * 2.5)))
        ds_f = min(ds_f, n_t)  # safety

        if ds_f > 1:
            n_ds = (n_t // ds_f) * ds_f
            block = data[:n_ds].reshape(-1, ds_f, n_ch).mean(axis=1).astype(np.float32)
            fs_eff = fs / ds_f
        else:
            block = data.astype(np.float32)
            fs_eff = fs

        # Bandpass filter
        filtered = _bandpass(block, lo_hz, hi_hz, fs_eff)

        # Per-channel z-score
        ch_mean = filtered.mean(axis=0, keepdims=True)
        ch_std = filtered.std(axis=0, keepdims=True)
        ch_std = np.where(ch_std == 0, 1.0, ch_std)
        zsc = (filtered - ch_mean) / ch_std

        # Remove common-mode (spatial median per time step)
        common = np.median(zsc, axis=1, keepdims=True)
        wf = zsc - common

        duration_s = wf.shape[0] / fs_eff
        rms_per_ch = np.sqrt(np.mean(wf ** 2, axis=0))

        # Downsample waterfall for HTML
        t_step = max(1, wf.shape[0] // max_t)
        ch_step = max(1, wf.shape[1] // max_ch)
        wf_ds = wf[::t_step, ::ch_step]

        band_results.append({
            "band_name": band_name,
            "lo_hz": lo_hz, "hi_hz": hi_hz,
            "waterfall": wf_ds.tolist(),
            "duration_s": float(duration_s),
            "n_t_ds": wf_ds.shape[0],
            "n_ch_ds": wf_ds.shape[1],
            "t_step": t_step,
            "ch_step": ch_step,
            "fs_eff": float(fs_eff),
            "rms_mean": float(np.mean(rms_per_ch)),
            "rms_max": float(np.max(rms_per_ch)),
        })

        del filtered, zsc, wf, block

    summary_parts = [f"{b['band_name']}: RMS={b['rms_mean']:.2f}" for b in band_results]

    return {
        "name": "Multi-Band Waterfall",
        "passed": True,  # informational
        "summary": f"{len(band_results)} bands analysed",
        "details": {
            "bands": band_results,
            "n_channels": n_ch,
            "duration_s": float(n_t / fs),
        },
    }


# ── Check 9: Wake-line analysis (DAShip) ────────────────────────────────

def check_wake_energy(data: np.ndarray, fs: float, dx: float = 10.0,
                      ds_target_hz: float = 10.0,
                      wake_band: tuple = (0.01, 1.0),
                      v_min: float = 4.0, v_max: float = 10.0) -> dict:
    """DAShip-inspired wake-line analysis (Huang et al., IEEE JSTARS 2025).

    Pipeline: downsample → z-score → bandpass → f-k velocity filter.
    Informational — produces waterfalls for visual inspection.
    """
    n_t, n_ch = data.shape if data.ndim == 2 else (len(data), 1)
    if n_ch < 10 or n_t < 800:
        return {
            "name": "Wake-Line Analysis (DAShip)",
            "passed": True,
            "summary": "Insufficient data for wake analysis",
            "details": {"n_frames": n_t, "n_channels": n_ch},
        }

    # Downsample
    ds_factor = max(1, int(round(fs / ds_target_hz)))
    fs_ds = fs / ds_factor
    n_ds = (n_t // ds_factor) * ds_factor
    ds = data[:n_ds].reshape(-1, ds_factor, n_ch).mean(axis=1).astype(np.float64)

    # Z-score normalization
    mu = np.mean(ds, axis=0, keepdims=True)
    sigma = np.std(ds, axis=0, keepdims=True)
    sigma = np.where(sigma == 0, 1.0, sigma)
    ds_z = (ds - mu) / sigma

    # Bandpass
    wake_filtered = _bandpass(ds_z, wake_band[0], wake_band[1], fs_ds)

    # Per-channel wake RMS
    wake_rms = np.sqrt(np.mean(wake_filtered ** 2, axis=0))

    # f-k velocity filter
    nt_ds, nx_ds = ds_z.shape
    t_cap = min(nt_ds, 4096)
    fk_block = wake_filtered[:t_cap, :]
    data_fk = fftshift(fft2(fk_block))
    f_axis = fftshift(fftfreq(t_cap, d=1.0 / fs_ds))
    k_axis = fftshift(fftfreq(nx_ds, d=dx))
    K, F = np.meshgrid(k_axis, f_axis)
    K_safe = np.where(K == 0, 1e-10, K)
    V = F / K_safe
    mask = (np.abs(V) >= v_min) & (np.abs(V) <= v_max)
    fk_filtered = np.real(ifft2(ifftshift(data_fk * mask)))
    fk_rms = np.sqrt(np.mean(fk_filtered ** 2, axis=0))

    # Metrics
    med_wake = float(np.median(wake_rms))
    wake_contrast = wake_rms / max(med_wake, 1e-10)
    peak_contrast = float(np.max(wake_contrast))

    # Downsample for report — keep higher resolution (up to 800 x 700)
    step = max(1, n_ch // 700)
    t_step = max(1, nt_ds // 800)
    ch_step = max(1, n_ch // 700)

    # Raw downsampled (before Z-score) — shows absolute amplitude
    ds_raw = ds  # mean-downsampled, not Z-scored

    return {
        "name": "Wake-Line Analysis (DAShip)",
        "passed": True,
        "summary": (f"Spatial contrast: {peak_contrast:.1f}x median — "
                    f"see waterfalls for visual inspection"),
        "details": {
            "ds_factor": ds_factor,
            "fs_downsampled_hz": fs_ds,
            "wake_band_hz": list(wake_band),
            "velocity_filter_ms": [v_min, v_max],
            "peak_spatial_contrast": peak_contrast,
            "median_wake_rms": med_wake,
            "wake_rms": wake_rms[::step].tolist(),
            "fk_rms": fk_rms[::step].tolist(),
            "channels": list(range(0, n_ch, step)),
            "n_channels": n_ch,
            "duration_s": float(n_t / fs),
            "waterfall_raw": ds_raw[::t_step, ::ch_step].tolist(),
            "waterfall_ds": ds_z[::t_step, ::ch_step].tolist(),
            "waterfall_wake": wake_filtered[::t_step, ::ch_step].tolist(),
            "waterfall_fk": fk_filtered[::max(1, t_cap // 800), ::ch_step].tolist(),
            "waterfall_t_step": t_step,
            "waterfall_ch_step": ch_step,
        },
    }
