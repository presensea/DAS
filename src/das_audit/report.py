"""
HTML report generator for das-audit.

Produces a self-contained interactive HTML file with Plotly.js charts.
All plots are embedded as JSON and rendered client-side.
"""

import json
from datetime import datetime
from typing import List

import numpy as np


# ── Plotly figure builders ─────────────────────────────────────────────

def _fig_channel_rms(details: dict) -> dict:
    """Per-channel RMS log-scale line plot."""
    rms = details["rms"]
    n = len(rms)
    traces = [{
        "x": list(range(n)),
        "y": rms,
        "type": "scatter",
        "mode": "lines",
        "line": {"width": 0.8, "color": "#2563eb"},
        "name": "RMS",
    }]
    # Median line
    traces.append({
        "x": [0, n - 1],
        "y": [details["median_rms"]] * 2,
        "type": "scatter",
        "mode": "lines",
        "line": {"dash": "dash", "color": "orange", "width": 1.5},
        "name": f"Median={details['median_rms']:.2e}",
    })
    if details["dead_channels"]:
        traces.append({
            "x": details["dead_channels"],
            "y": [min(r for r in rms if r > 0) if any(r > 0 for r in rms) else 1e-10] * len(details["dead_channels"]),
            "type": "scatter",
            "mode": "markers",
            "marker": {"color": "red", "size": 5},
            "name": f"Dead ({details['n_dead']})",
        })
    if details["spike_channels"]:
        traces.append({
            "x": details["spike_channels"],
            "y": [rms[i] for i in details["spike_channels"]],
            "type": "scatter",
            "mode": "markers",
            "marker": {"color": "magenta", "size": 5},
            "name": f"Saturated ({details['n_spike']})",
        })
    return {
        "data": traces,
        "layout": {
            "title": "Per-Channel RMS",
            "xaxis": {"title": "Channel"},
            "yaxis": {"title": "RMS", "type": "log"},
            "height": 300,
            "margin": {"t": 40, "b": 40, "l": 60, "r": 20},
            "legend": {"x": 1, "xanchor": "right", "y": 1},
        },
    }


def _fig_spectral(details: dict) -> list:
    """Spectral baseline: PSD curves + per-channel 50Hz power heatmap."""
    figs = []

    # 1) PSD curves for sampled channels
    if details.get("freqs") and details.get("psd_matrix"):
        freqs = details["freqs"]
        traces = []
        for i, psd in enumerate(details["psd_matrix"]):
            traces.append({
                "x": freqs,
                "y": psd,
                "type": "scatter",
                "mode": "lines",
                "line": {"width": 0.7},
                "name": f"ch {details['psd_channels'][i]}",
            })
        for peak in details.get("noise_peaks", [])[:10]:
            traces.append({
                "x": [peak["freq_hz"]] * 2,
                "y": [min(psd) for psd in details["psd_matrix"]] + [max(max(p) for p in details["psd_matrix"])],
                "type": "scatter",
                "mode": "lines",
                "line": {"dash": "dot", "color": "red", "width": 1},
                "showlegend": False,
            })
        figs.append({
            "data": traces,
            "layout": {
                "title": "Spectral Baseline (Welch PSD)",
                "xaxis": {"title": "Frequency (Hz)"},
                "yaxis": {"title": "PSD", "type": "log"},
                "height": 300,
                "margin": {"t": 40, "b": 40, "l": 60, "r": 20},
            },
        })

    # 2) Per-channel 50 Hz power ratio map
    if details.get("scan_channels") and details.get("power_50hz"):
        chs = details["scan_channels"]
        figs.append({
            "data": [{
                "x": chs,
                "y": details["power_50hz"],
                "type": "bar",
                "marker": {
                    "color": ["#ef4444" if v > 5 else "#94a3b8" for v in details["power_50hz"]],
                },
                "name": "50 Hz ratio",
            }, {
                "x": [chs[0], chs[-1]],
                "y": [5, 5],
                "type": "scatter",
                "mode": "lines",
                "line": {"dash": "dash", "color": "red", "width": 1.5},
                "name": "Threshold (5x)",
            }],
            "layout": {
                "title": f"50 Hz Power Ratio per Channel ({details.get('n_channels_with_50hz', 0)} channels above threshold)",
                "xaxis": {"title": "Channel"},
                "yaxis": {"title": "Power ratio (peak / local median)"},
                "height": 280,
                "margin": {"t": 40, "b": 40, "l": 60, "r": 20},
                "showlegend": True,
            },
        })

    return figs


def _fig_coupling(details: dict) -> list:
    """Coupling coherence: raw NCC + rolling median + segment shading."""
    figs = []
    channels = details["channels"]
    ncc = details["ncc"]
    rolling = details.get("rolling_ncc", [])
    segments = details.get("segments", [])

    # Main NCC plot with rolling median and segment bands
    traces = [{
        "x": channels,
        "y": ncc,
        "type": "scatter",
        "mode": "lines",
        "line": {"width": 0.6, "color": "#94a3b8"},
        "name": "NCC (raw)",
    }]
    if rolling:
        traces.append({
            "x": channels,
            "y": rolling,
            "type": "scatter",
            "mode": "lines",
            "line": {"width": 2, "color": "#2563eb"},
            "name": "Rolling median",
        })
    # Threshold lines
    for thr, label, color in [(0.3, "Aerial/Sub", "#f59e0b"), (0.6, "Buried", "#22c55e")]:
        traces.append({
            "x": [channels[0], channels[-1]],
            "y": [thr, thr],
            "type": "scatter",
            "mode": "lines",
            "line": {"dash": "dash", "color": color, "width": 1},
            "name": f"{label} threshold ({thr})",
        })

    # Segment shading
    seg_colors = {"aerial/submarine": "rgba(239,68,68,0.1)",
                  "conduit/loosely-coupled": "rgba(245,158,11,0.1)",
                  "buried/ground-coupled": "rgba(34,197,94,0.1)"}
    shapes = []
    for seg in segments:
        shapes.append({
            "type": "rect",
            "x0": seg["ch_start"], "x1": seg["ch_end"],
            "y0": -0.2, "y1": 1.05,
            "fillcolor": seg_colors.get(seg["type"], "rgba(0,0,0,0.05)"),
            "line": {"width": 0},
            "layer": "below",
        })

    figs.append({
        "data": traces,
        "layout": {
            "title": f"Coupling Coherence (median NCC={details['median_ncc']:.3f} → {details.get('overall_type', '?')})",
            "xaxis": {"title": "Channel"},
            "yaxis": {"title": "NCC(k, k+1)", "range": [-0.2, 1.05]},
            "height": 350,
            "margin": {"t": 40, "b": 40, "l": 60, "r": 20},
            "shapes": shapes,
        },
    })

    return figs


def _fig_time_gaps(details: dict) -> dict:
    """Time gap bar chart."""
    if details.get("n_gaps", 0) == 0:
        return None
    return {
        "data": [{
            "x": details["gap_indices"],
            "y": details["gap_sizes_s"],
            "type": "bar",
            "marker": {"color": "#ef4444"},
        }],
        "layout": {
            "title": f"Time Gaps ({details['n_gaps']} gaps, {details['total_missing_s']:.1f}s missing)",
            "xaxis": {"title": "Sample index"},
            "yaxis": {"title": "Gap size (s)"},
            "height": 250,
            "margin": {"t": 40, "b": 40, "l": 60, "r": 20},
        },
    }


def _fig_multiband_waterfall(details: dict) -> list:
    """Multi-band waterfall: signed bandpass heatmaps (z-score, common-mode removed)."""
    figs = []
    bands = details.get("bands", [])
    n_ch_total = details.get("n_channels", 1)

    if not bands:
        return figs

    for band in bands:
        wf = band.get("waterfall", [])
        if not wf:
            continue
        n_rows = len(wf)
        n_cols = len(wf[0]) if wf else 0
        dur = band.get("duration_s", 1)
        ch_step = band.get("ch_step", 1)

        y_arr = [round(r * dur / n_rows, 2) for r in range(n_rows)]
        x_arr = [c * ch_step for c in range(n_cols)]

        # Symmetric color range from data
        flat = [v for row in wf for v in row]
        if flat:
            import statistics
            sorted_abs = sorted(abs(v) for v in flat)
            p99 = sorted_abs[int(len(sorted_abs) * 0.99)] if sorted_abs else 1
        else:
            p99 = 1

        figs.append({
            "data": [{
                "z": wf,
                "x": x_arr,
                "y": y_arr,
                "type": "heatmap",
                "colorscale": [
                    [0.0, "#053061"], [0.15, "#2166ac"], [0.3, "#4393c3"],
                    [0.45, "#d1e5f0"], [0.5, "#f7f7f7"],
                    [0.55, "#fddbc7"], [0.7, "#f4a582"],
                    [0.85, "#d6604d"], [1.0, "#67001f"],
                ],
                "zmin": -p99,
                "zmid": 0,
                "zmax": p99,
                "colorbar": {"title": "z-score"},
            }],
            "layout": {
                "title": f"{band['band_name']} — Bandpass Waterfall (z-score, common-mode removed)",
                "xaxis": {"title": "Channel"},
                "yaxis": {"title": "Time (s)"},
                "height": 500,
                "margin": {"t": 50, "b": 40, "l": 60, "r": 80},
            },
        })

    return figs


def _fig_wake(details: dict) -> list:
    """Wake analysis waterfalls + RMS profiles."""
    figs = []
    channels = details.get("channels", [])
    if not channels:
        return figs

    n_ch = details["n_channels"]
    dur = details["duration_s"]
    ch_step = details.get("waterfall_ch_step", 1)

    # Waterfalls — use explicit x/y arrays for correct axis mapping
    waterfalls = [
        ("waterfall_raw", f"Raw Downsampled ({details['fs_downsampled_hz']:.0f} Hz)", "Viridis"),
        ("waterfall_ds", f"Z-score Normalised ({details['fs_downsampled_hz']:.0f} Hz)", "RdBu_r"),
        ("waterfall_wake", f"Wake Band ({details['wake_band_hz'][0]}-{details['wake_band_hz'][1]} Hz)", "RdBu_r"),
        ("waterfall_fk", f"f-k Filtered ({details['velocity_filter_ms'][0]}-{details['velocity_filter_ms'][1]} m/s)", "RdBu_r"),
    ]
    for wf_key, title, colorscale in waterfalls:
        wf = details.get(wf_key)
        if not wf:
            continue
        wf_arr = np.array(wf)
        n_rows, n_cols = wf_arr.shape
        lo, hi = float(np.percentile(wf_arr, 1)), float(np.percentile(wf_arr, 99))

        x_arr = [int(c * ch_step) for c in range(n_cols)]
        y_arr = [round(r * dur / n_rows, 2) for r in range(n_rows)]

        figs.append({
            "data": [{
                "z": wf,
                "x": x_arr,
                "y": y_arr,
                "type": "heatmap",
                "colorscale": colorscale,
                "zmin": lo,
                "zmax": hi,
                "colorbar": {"title": "Amp"},
            }],
            "layout": {
                "title": title,
                "xaxis": {"title": "Channel"},
                "yaxis": {"title": "Time (s)"},
                "height": 450,
                "margin": {"t": 40, "b": 40, "l": 60, "r": 80},
            },
        })

    # RMS profiles
    wake_rms = details.get("wake_rms", [])
    fk_rms = details.get("fk_rms", [])
    if wake_rms:
        figs.append({
            "data": [{
                "x": channels,
                "y": wake_rms,
                "type": "scatter",
                "mode": "lines",
                "line": {"width": 1, "color": "#2563eb"},
                "name": "Wake band RMS",
            }, {
                "x": channels,
                "y": fk_rms,
                "type": "scatter",
                "mode": "lines",
                "line": {"width": 1, "color": "#dc2626"},
                "name": "f-k filtered RMS",
            }],
            "layout": {
                "title": "Wake & f-k Filtered RMS Profile",
                "xaxis": {"title": "Channel"},
                "yaxis": {"title": "RMS (Z-score units)"},
                "height": 300,
                "margin": {"t": 40, "b": 40, "l": 60, "r": 20},
            },
        })

    # Spatial contrast
    contrast = details.get("wake_contrast", [])
    if contrast:
        figs.append({
            "data": [{
                "x": channels,
                "y": contrast,
                "type": "bar",
                "marker": {"color": ["#f59e0b" if c > 1.5 else "#94a3b8" for c in contrast]},
                "name": "Spatial contrast",
            }],
            "layout": {
                "title": "Spatial Contrast (RMS / median RMS)",
                "xaxis": {"title": "Channel"},
                "yaxis": {"title": "Contrast ratio"},
                "height": 280,
                "margin": {"t": 40, "b": 40, "l": 60, "r": 20},
            },
        })

    return figs


# ── Check explanations ──────────────────────────────────────────────────
CHECK_EXPLANATIONS = {
    "Metadata Validation": {
        True: ("All required header fields (pulse_width, rep_frequency, points_per_frame, "
               "num_frames) are present and non-zero. The gauge length and output rate are plausible."),
        False: ("One or more required header fields are missing or zero. This can indicate "
                "a corrupted file, an incomplete recording, or a firmware version that omits "
                "certain fields (e.g. gauge_length=0 is common on older DistraSense units)."),
    },
    "Saturation Check": {
        True: ("Less than 0.1% of samples are NaN or clipped at the ADC rail. "
               "The signal is within the dynamic range of the digitizer."),
        False: ("More than 0.1% of samples are either NaN or clipped at the "
                "min/max of the data type range. Clipped regions lose phase information."),
    },
    "Per-Channel RMS": {
        True: ("All channels show non-zero, non-anomalous RMS levels."),
        "warn": ("Only edge channels are dead or anomalous. Edge channels "
                 "commonly show artifacts from the DAS interrogator's measurement window "
                 "boundary. Exclude these from analysis."),
        False: ("Some channels have zero RMS (dead — fiber break) or anomalously high RMS "
                "(>10x median — noisy connector, laser reflection, or damaged fiber section)."),
    },
    "Spectral Baseline": {
        True: ("No dominant peaks at 50/60 Hz mains frequency. The cable is "
               "free of electrical interference."),
        False: ("Peaks detected at 50/60 Hz on some channels, indicating electromagnetic "
                "interference from mains power. These should be notch-filtered before analysis."),
    },
    "Time Continuity": {
        True: ("No gaps detected. All frames arrived at the expected rate."),
        False: ("Gaps >1.5x the expected interval found — dropped frames or acquisition interrupts."),
    },
    "Coupling Coherence": {
        True: ("Most adjacent channel pairs show NCC above 0.3. Good mechanical coupling."),
        "warn": ("A large fraction of channels have NCC below 0.3. "
                 "Cable segments classified as: "
                 "<b>buried</b> (NCC >= 0.6), <b>conduit</b> (0.3-0.6), <b>aerial/submarine</b> (< 0.3). "
                 "Low NCC is expected for aerial and submarine cable deployments."),
    },
    "f-k Aliasing": {
        True: ("Energy concentrated away from spatial Nyquist. Channel spacing adequately samples the wavefield."),
        False: ("Significant energy at spatial Nyquist boundary. Some f-k aliasing is expected at ~10 m spacing."),
    },
    "Multi-Band Waterfall": {
        True: ("Per-band bandpass waterfalls (z-score normalised, common-mode removed). "
               "Diagonal features indicate propagating signals (waves, wakes, Scholte waves)."),
    },
    "Wake-Line Analysis (DAShip)": {
        True: ("DAShip pipeline (Huang et al., IEEE JSTARS 2025): downsample to 10 Hz, "
               "Z-score normalize, bandpass 0.01-1 Hz, f-k filter 4-10 m/s. "
               "Inspect waterfalls visually for diagonal wake signatures."),
        False: ("Insufficient data for DAShip wake analysis."),
    },
}


def _status_badge(chk: dict) -> str:
    status = chk.get("status", "pass" if chk["passed"] else "fail")
    if status == "warn":
        return '<span style="background:#f59e0b;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;">WARN</span>'
    elif status == "pass" or (status not in ("fail", "warn") and chk["passed"]):
        return '<span style="background:#22c55e;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;">PASS</span>'
    return '<span style="background:#ef4444;color:white;padding:2px 8px;border-radius:4px;font-weight:bold;">FAIL</span>'


def _fig_to_div(fig_json: dict, div_id: str) -> str:
    """Convert a Plotly figure dict to an HTML div + JS snippet."""
    return (
        f'<div id="{div_id}" style="width:100%;max-width:960px;"></div>\n'
        f'<script>Plotly.newPlot("{div_id}", {json.dumps(fig_json["data"], default=str)}, '
        f'{json.dumps(fig_json["layout"], default=str)}, '
        f'{{"responsive": true, "displayModeBar": true, "modeBarButtonsToRemove": ["lasso2d","select2d"]}});</script>\n'
    )


def generate_html_report(checks: List[dict], metadata: dict, output_path: str) -> str:
    """Generate a self-contained interactive HTML report with Plotly plots."""

    # Build plots for each check
    plot_divs = {}  # check name -> HTML string
    div_counter = 0

    for chk in checks:
        name = chk["name"]
        html_parts = []

        if name == "Per-Channel RMS":
            fig = _fig_channel_rms(chk["details"])
            html_parts.append(_fig_to_div(fig, f"plot_{div_counter}"))
            div_counter += 1

        elif name == "Spectral Baseline":
            for fig in _fig_spectral(chk["details"]):
                html_parts.append(_fig_to_div(fig, f"plot_{div_counter}"))
                div_counter += 1

        elif name == "Coupling Coherence":
            for fig in _fig_coupling(chk["details"]):
                html_parts.append(_fig_to_div(fig, f"plot_{div_counter}"))
                div_counter += 1

        elif name == "Time Continuity":
            fig = _fig_time_gaps(chk["details"])
            if fig:
                html_parts.append(_fig_to_div(fig, f"plot_{div_counter}"))
                div_counter += 1

        elif name == "Multi-Band Waterfall":
            for fig in _fig_multiband_waterfall(chk["details"]):
                html_parts.append(_fig_to_div(fig, f"plot_{div_counter}"))
                div_counter += 1

        elif name == "Wake-Line Analysis (DAShip)":
            for fig in _fig_wake(chk["details"]):
                html_parts.append(_fig_to_div(fig, f"plot_{div_counter}"))
                div_counter += 1

        if html_parts:
            plot_divs[name] = "\n".join(html_parts)

    # Compute overall status
    n_total = len(checks)
    n_fail = sum(1 for c in checks if c.get("status", "pass" if c["passed"] else "fail") == "fail")
    n_warn = sum(1 for c in checks if c.get("status") == "warn")
    n_pass = n_total - n_fail - n_warn
    if n_fail == 0 and n_warn == 0:
        overall = "ALL PASS"
        overall_color = "#22c55e"
    elif n_fail == 0:
        overall = f"{n_pass} PASS, {n_warn} WARN"
        overall_color = "#f59e0b"
    else:
        overall = f"{n_pass} PASS, {n_warn} WARN, {n_fail} FAIL"
        overall_color = "#ef4444" if n_pass <= n_total // 2 else "#f59e0b"

    # Build check cards
    check_rows = ""
    for chk in checks:
        plots_html = plot_divs.get(chk["name"], "")

        # Exclude bulky arrays from JSON dump
        skip_keys = {"rms", "ncc", "rolling_ncc", "psd_matrix",
                     "power_50hz", "power_60hz", "freqs"}
        details_for_json = {}
        for k, v in chk["details"].items():
            if k.startswith("waterfall_") or k in skip_keys:
                continue
            # For band results, strip the waterfall arrays
            if k == "bands" and isinstance(v, list):
                clean_bands = []
                for b in v:
                    clean_bands.append({bk: bv for bk, bv in b.items()
                                        if bk not in ("waterfall", "ncc_waterfall",
                                                       "median_per_channel")})
                details_for_json[k] = clean_bands
                continue
            details_for_json[k] = v
        details_json = json.dumps(details_for_json, indent=2, default=str)

        # Explanation
        expl_map = CHECK_EXPLANATIONS.get(chk["name"], {})
        status_key = chk.get("status")
        explanation = expl_map.get(status_key, expl_map.get(chk["passed"], ""))
        expl_html = f'<div class="explanation">{explanation}</div>' if explanation else ""

        check_rows += f"""
        <div class="check-card">
            <div class="check-header">
                {_status_badge(chk)}
                <strong>{chk['name']}</strong>
                <span class="check-summary">{chk['summary']}</span>
            </div>
            {expl_html}
            {plots_html}
            <details><summary>Raw details (JSON)</summary><pre>{details_json}</pre></details>
        </div>
        """

    # Metadata table
    meta_rows = ""
    for k, v in metadata.items():
        meta_rows += f"<tr><td><strong>{k}</strong></td><td>{v}</td></tr>"

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DAS Audit Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f8fafc; color: #1e293b; }}
  .container {{ max-width: 1020px; margin: 0 auto; }}
  h1 {{ color: #0f172a; border-bottom: 3px solid #2563eb; padding-bottom: 8px; }}
  .overall {{ font-size: 1.5em; padding: 16px; border-radius: 8px; background: {overall_color}; color: white; text-align: center; margin: 16px 0; }}
  .check-card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; margin: 12px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .check-header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; flex-wrap: wrap; }}
  .check-summary {{ color: #64748b; font-size: 0.9em; }}
  .explanation {{ background: #f8fafc; border-left: 3px solid #2563eb; padding: 8px 12px; margin: 8px 0; font-size: 0.88em; color: #334155; line-height: 1.5; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 6px 10px; text-align: left; font-size: 0.9em; }}
  th {{ background: #f1f5f9; }}
  pre {{ background: #f1f5f9; padding: 8px; border-radius: 4px; font-size: 0.8em; overflow-x: auto; max-height: 200px; }}
  details {{ margin-top: 8px; }}
  summary {{ cursor: pointer; color: #2563eb; font-size: 0.85em; }}
  .footer {{ text-align: center; color: #94a3b8; font-size: 0.8em; margin-top: 32px; }}
</style>
</head>
<body>
<div class="container">
  <h1>DAS Data Quality Audit</h1>
  <div class="overall">Overall: {overall}</div>

  <h2>Dataset Metadata</h2>
  <table>{meta_rows}</table>

  <h2>Quality Checks</h2>
  {check_rows}

  <div class="footer">
    Generated by <strong>das-audit v0.2.0</strong> on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </div>
</div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return output_path
