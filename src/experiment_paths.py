from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


DEFAULT_EXPERIMENT_VERSION = "LATEST"


def normalize_version(version: str | None) -> str:
    if version is None:
        return DEFAULT_EXPERIMENT_VERSION
    text = version.strip()
    return text.upper() if text else DEFAULT_EXPERIMENT_VERSION


def versioned_dir(project_root: Path, category: str, version: str) -> Path:
    return project_root / category / normalize_version(version)


def log_dir(project_root: Path, version: str) -> Path:
    return versioned_dir(project_root, "records", version) / "logs"


def record_dir(project_root: Path, version: str) -> Path:
    return versioned_dir(project_root, "records", version)


def run_dir(project_root: Path, version: str) -> Path:
    return versioned_dir(project_root, "records", version) / "runs"


def checkpoint_dir(project_root: Path, version: str) -> Path:
    return versioned_dir(project_root, "records", version) / "checkpoints"


def analysis_dir(project_root: Path, version: str) -> Path:
    return versioned_dir(project_root, "records", version) / "analysis"


def manifest_payload(project_root: Path) -> Dict[str, object]:
    version = DEFAULT_EXPERIMENT_VERSION
    return {
        "default_version": version,
        "versions": {
            version: {
                "records_dir": str(record_dir(project_root, version).relative_to(project_root)),
                "logs_dir": str(log_dir(project_root, version).relative_to(project_root)),
                "runs_dir": str(run_dir(project_root, version).relative_to(project_root)),
                "checkpoints_dir": str(checkpoint_dir(project_root, version).relative_to(project_root)),
                "analysis_dir": str(analysis_dir(project_root, version).relative_to(project_root)),
            }
        },
    }


def ensure_version_manifest(project_root: Path) -> Path:
    manifest = project_root / "records" / "experiment_versions.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(manifest_payload(project_root), indent=2), encoding="utf-8")
    return manifest
