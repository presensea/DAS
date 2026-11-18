"""
Provenance logging: write every QA pass to a Parquet database for historical trust-scoring.
"""

import os
import json
from datetime import datetime
from typing import List

import numpy as np


def log_audit(checks: List[dict], metadata: dict, dataset_name: str,
              db_path: str = "das_audit_provenance.parquet") -> str:
    """Append audit results to a Parquet provenance database."""
    try:
        import pandas as pd
    except ImportError:
        # Fallback to JSON lines if pandas not available
        return _log_audit_jsonl(checks, metadata, dataset_name, db_path.replace(".parquet", ".jsonl"))

    timestamp = datetime.now().isoformat()
    rows = []

    for chk in checks:
        row = {
            "timestamp": timestamp,
            "dataset": dataset_name,
            "check_name": chk["name"],
            "passed": chk["passed"],
            "summary": chk["summary"],
        }
        # Flatten key details
        for k, v in chk["details"].items():
            if isinstance(v, (int, float, str, bool)):
                row[f"detail_{k}"] = v
        rows.append(row)

    df_new = pd.DataFrame(rows)

    if os.path.exists(db_path):
        df_old = pd.read_parquet(db_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    # Cast mixed-type detail columns to string to avoid parquet errors
    for col in df.columns:
        if col.startswith("detail_") and df[col].dtype == object:
            df[col] = df[col].astype(str)

    df.to_parquet(db_path, index=False)
    return db_path


def _log_audit_jsonl(checks: List[dict], metadata: dict, dataset_name: str,
                     db_path: str) -> str:
    """Fallback: append audit as JSON lines."""
    timestamp = datetime.now().isoformat()
    with open(db_path, "a") as f:
        for chk in checks:
            record = {
                "timestamp": timestamp,
                "dataset": dataset_name,
                "check_name": chk["name"],
                "passed": chk["passed"],
                "summary": chk["summary"],
            }
            f.write(json.dumps(record) + "\n")
    return db_path
