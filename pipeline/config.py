"""
pipeline/config.py
==================
YAML 설정 파일 로드, deep merge, dot-path 오버라이드 지원.
"""

import copy
import ast
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def set_nested(cfg: dict, dot_path: str, value) -> None:
    """'feature_extraction.dwt.level' 형식의 경로로 값을 설정."""
    keys = dot_path.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def get_nested(cfg: dict, dot_path: str):
    """dot-path로 값을 읽어 반환."""
    keys = dot_path.split(".")
    d = cfg
    for k in keys:
        d = d[k]
    return d


def parse_value(raw: str):
    """CLI --set 에서 넘어온 문자열 값을 Python 타입으로 변환."""
    try:
        return ast.literal_eval(raw)
    except Exception:
        # bool 문자열 처리
        if raw.lower() == "true":
            return True
        if raw.lower() == "false":
            return False
        return raw


def validate_config(cfg: dict) -> None:
    required = ["experiment", "feature_extraction", "learning"]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config missing required key: '{key}'")

    method = cfg["feature_extraction"].get("method", "")
    valid = ["dwt", "dfs", "sdp", "tddfs"]
    if method not in valid:
        raise ValueError(f"Unknown feature method '{method}'. Choose from {valid}")

    sw = cfg.get("sliding_window", {})
    if sw.get("enabled", False):
        w = sw.get("window_sec", 1.0)
        h = sw.get("hop_sec", 0.5)
        if not (0.5 <= w <= 5.0):
            raise ValueError(f"sliding_window.window_sec={w} 범위 초과 (0.5~5.0)")
        if h >= w:
            raise ValueError(f"hop_sec({h}) >= window_sec({w}) — hop이 너무 큼")
