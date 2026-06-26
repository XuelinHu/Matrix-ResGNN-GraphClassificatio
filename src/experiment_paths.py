"""统一管理实验版本、记录目录、日志目录、检查点目录和分析目录路径。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


# 默认实验版本：所有记录默认写入 records/LATEST。
DEFAULT_EXPERIMENT_VERSION = "LATEST"


def normalize_version(version: str | None) -> str:
    """规范化实验版本名，空值时回退到默认版本。"""
    if version is None:
        return DEFAULT_EXPERIMENT_VERSION
    text = version.strip()
    return text.upper() if text else DEFAULT_EXPERIMENT_VERSION


def versioned_dir(project_root: Path, category: str, version: str) -> Path:
    """构造指定类别和版本对应的根目录。"""
    return project_root / category / normalize_version(version)


def log_dir(project_root: Path, version: str) -> Path:
    """返回当前版本的 result JSON 日志目录。"""
    return versioned_dir(project_root, "records", version) / "logs"


def record_dir(project_root: Path, version: str) -> Path:
    """返回当前版本的 records 根目录。"""
    return versioned_dir(project_root, "records", version)


def run_dir(project_root: Path, version: str) -> Path:
    """返回当前版本的运行脚本和队列记录目录。"""
    return versioned_dir(project_root, "records", version) / "runs"


def checkpoint_dir(project_root: Path, version: str) -> Path:
    """返回当前版本的模型检查点目录。"""
    return versioned_dir(project_root, "records", version) / "checkpoints"


def analysis_dir(project_root: Path, version: str) -> Path:
    """返回当前版本的机制分析和中间产物目录。"""
    return versioned_dir(project_root, "records", version) / "analysis"


def manifest_payload(project_root: Path) -> Dict[str, object]:
    """生成实验版本 manifest 的 JSON 内容。"""
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
    """确保实验版本 manifest 文件存在并写入最新路径信息。"""
    manifest = project_root / "records" / "experiment_versions.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(manifest_payload(project_root), indent=2), encoding="utf-8")
    return manifest
