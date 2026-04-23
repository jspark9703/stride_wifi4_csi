"""
pipeline/runner.py
===================
YAML 설정 → 기존 모듈 함수 호출 오케스트레이터.
기존 코드(extract_dwt.py, dwt_mlp.py 등)는 수정 없이 그대로 사용.
"""

import os
import sys
import json
import asyncio
import types
from datetime import datetime

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ══════════════════════════════════════════════════════════════
#  디렉토리 구성
# ══════════════════════════════════════════════════════════════

def get_dirs(cfg: dict) -> dict:
    date_tag = cfg["experiment"].get("date_tag", "")
    exp_name = cfg["experiment"]["name"]
    method   = cfg["feature_extraction"]["method"]
    steps    = cfg["experiment"].get("steps", ["extract", "train"])

    sweep_name = cfg["experiment"].get("_sweep_name")
    sweep_ts   = cfg["experiment"].get("_sweep_ts")

    # Baseline sanitization 경로 (참조/재사용)
    baseline_sanit = os.path.join(BASE_DIR, "data", "sanitization", "sanitization", date_tag)

    # sanitize step 없으면 baseline sanitization 재사용
    sanit_is_reused = "sanitize" not in steps

    sw = cfg.get("sliding_window", {})

    if sweep_name and sweep_ts:
        # ── Ablation mode
        abl_root  = os.path.join(BASE_DIR, "data", "ablation", f"{sweep_ts}_{sweep_name}", exp_name)
        res_root  = os.path.join(BASE_DIR, "results", f"{sweep_ts}_{sweep_name}")

        sanit     = baseline_sanit if sanit_is_reused else os.path.join(abl_root, "sanitization")
        feat      = os.path.join(abl_root, "feature_extraction")
        res_learn = os.path.join(res_root, "learning", exp_name)
        res_feat  = os.path.join(res_root, "feature_extraction", exp_name)
        models    = os.path.join(res_root, "learning", exp_name, "models")

    elif exp_name == "baseline":
        # ── Baseline single run mode
        sanit     = baseline_sanit
        feat      = os.path.join(BASE_DIR, "data", "feature_extraction", method, "baseline")
        res_root  = os.path.join(BASE_DIR, "results", "baseline")
        res_learn = os.path.join(res_root, "learning")
        res_feat  = os.path.join(res_root, "feature_extraction")
        models    = os.path.join(res_root, "learning", "models")

    else:
        # ── Other single run mode
        sanit     = baseline_sanit if sanit_is_reused else os.path.join(BASE_DIR, "data", "sanitization", "sanitization", date_tag)
        feat      = os.path.join(BASE_DIR, "data", "feature_extraction", method, exp_name)
        res_root  = os.path.join(BASE_DIR, "results", exp_name)
        res_learn = os.path.join(res_root, "learning")
        res_feat  = os.path.join(res_root, "feature_extraction")
        models    = os.path.join(res_root, "learning", "models")

    dirs = {
        "raw":      os.path.join(BASE_DIR, "data", "raw", date_tag),
        "prep":     os.path.join(BASE_DIR, "data", "sanitization", "preprocessed", date_tag),
        "sanit":    sanit,
        "json":     os.path.join(BASE_DIR, "data", "result", "sanitization", "json"),
        "plots":    os.path.join(BASE_DIR, "data", "result", "sanitization", "plots"),
        "log_win":  os.path.join(BASE_DIR, "data", "result", "windowing"),
        "feat":     feat,
        "res_feat": res_feat,
        "res_learn": res_learn,
        "models":   models,
    }

    # 슬라이딩 윈도우 활성화 시: 모든 단계(prep, sanit)가 이미 윈도우 단위임
    dirs["feat_input"] = dirs["sanit"]

    return dirs


# ══════════════════════════════════════════════════════════════
#  STEP 0 & 1: Preprocessing
# ══════════════════════════════════════════════════════════════

async def _step_preprocess(cfg: dict, dirs: dict) -> None:
    pcfg = cfg.get("preprocessing", {})

    _add_path(os.path.join(BASE_DIR, "src", "sanitization", "scripts"))
    import preprocess

    await preprocess.run_preprocessing(
        raw_dir           = dirs["raw"],
        out_dir           = dirs["prep"],
        json_dir          = dirs["json"],
        target_fs         = pcfg.get("target_fs",         100),
        max_gap_ms        = pcfg.get("max_gap_ms",         20.0),
        hampel_enabled    = pcfg.get("hampel_enabled",     True),
        hampel_window     = pcfg.get("hampel_window",      7),
        hampel_threshold  = pcfg.get("hampel_threshold",   5.0),
        lowpass_enabled   = pcfg.get("lowpass_enabled",    False),
        lowpass_cutoff    = pcfg.get("lowpass_cutoff",     11.0),
        window_sec        = cfg.get("sliding_window", {}).get("window_sec", 0) if cfg.get("sliding_window", {}).get("enabled", False) else 0,
        hop_sec           = cfg.get("sliding_window", {}).get("hop_sec", 0) if cfg.get("sliding_window", {}).get("enabled", False) else 0,
        run_id            = cfg["experiment"].get("run_id"),
    )


# ══════════════════════════════════════════════════════════════
#  STEP 2: Sanitization
# ══════════════════════════════════════════════════════════════

async def _step_sanitize(cfg: dict, dirs: dict) -> None:
    scfg = cfg.get("sanitization", {})

    _add_path(os.path.join(BASE_DIR, "src", "sanitization", "scripts"))

    # template 생성 (캘리브레이션 CSV 있을 때)
    if scfg.get("make_template", False):
        import make_template
        import numpy as np
        calib_dir    = os.path.join(BASE_DIR, scfg.get("calib_dir", "calibration"))
        template_out = os.path.join(BASE_DIR, scfg.get("template_csv",
                                    "src/sanitization/template/template.csv"))
        if os.path.isdir(calib_dir):
            make_template.run_make_template(
                calib_dir       = calib_dir,
                out_path        = template_out,
                linear_interval = np.arange(
                    scfg.get("linear_start", 30),
                    scfg.get("linear_end",   78),
                ),
                plot = True,
            )

    import sanitization as san_mod
    template_csv = os.path.join(BASE_DIR, scfg.get("template_csv",
                                "src/sanitization/template/template.csv"))

    await san_mod.run_sanitization(
        prep_dir     = dirs["prep"],
        sanit_dir    = dirs["sanit"],
        json_dir     = dirs["json"],
        enable_ratio = scfg.get("enable_ratio", False),
        run_id       = cfg["experiment"].get("run_id"),
        template_csv = template_csv if os.path.exists(template_csv) else None,
    )


# ══════════════════════════════════════════════════════════════
#  STEP W: Sliding Window
# ══════════════════════════════════════════════════════════════

def _step_window(cfg: dict, dirs: dict) -> None:
    from pipeline.windowing import apply_sliding_window

    sw  = cfg["sliding_window"]
    fs  = cfg.get("preprocessing", {}).get("target_fs", 100)

    apply_sliding_window(
        sanit_dir  = dirs["sanit"],
        out_dir    = dirs["windowed"],
        window_sec = sw.get("window_sec", 1.0),
        hop_sec    = sw.get("hop_sec",    0.5),
        fs         = fs,
        log_dir    = dirs["log_win"],
    )


# ══════════════════════════════════════════════════════════════
#  STEP 3: Feature Extraction
# ══════════════════════════════════════════════════════════════

def _step_extract(cfg: dict, dirs: dict) -> None:
    method = cfg["feature_extraction"]["method"]
    mcfg   = cfg["feature_extraction"].get(method, {})

    os.makedirs(dirs["feat"],     exist_ok=True)
    os.makedirs(dirs["res_feat"], exist_ok=True)

    if method == "dwt":
        _add_path(os.path.join(BASE_DIR, "src", "feature_extraction", "dwt"))
        import extract_dwt
        extract_dwt.run_dwt_extraction(
            sanit_dir = dirs["feat_input"],
            out_dir   = dirs["feat"],
            log_dir   = dirs["res_feat"],
            wavelet   = mcfg.get("wavelet", "sym3"),
            level     = mcfg.get("level",   10),
            n_pca     = mcfg.get("n_pca",   6),
        )

    elif method == "dfs":
        _add_path(os.path.join(BASE_DIR, "src", "feature_extraction", "dfs"))
        import extract_dfs
        extract_dfs.run_dfs_extraction(
            sanit_dir  = dirs["feat_input"],
            out_dir    = dirs["feat"],
            log_dir    = dirs["res_feat"],
            n_fft      = mcfg.get("n_fft",      64),
            hop        = mcfg.get("hop",          4),
            doppler_hz = mcfg.get("doppler_hz", 50.0),
        )

    elif method == "sdp":
        _add_path(os.path.join(BASE_DIR, "src", "feature_extraction", "sdp"))
        import extract_sdp
        extract_sdp.run_sdp_extraction(
            sanit_dir = dirs["feat_input"],
            out_dir   = dirs["feat"],
            log_dir   = dirs["res_feat"],
            n_lag     = mcfg.get("n_lag",     20),
            wt        = mcfg.get("wt",       100),
            hop       = mcfg.get("hop",       50),
            n_windows = mcfg.get("n_windows",  1),
        )

    elif method == "tddfs":
        _add_path(os.path.join(BASE_DIR, "src", "feature_extraction", "td-dfs"))
        import extract_tddfs
        extract_tddfs.run_tddfs_extraction(
            sanit_dir   = dirs["feat_input"],
            out_dir     = dirs["feat"],
            log_dir     = dirs["res_feat"],
            delta_t_min = mcfg.get("delta_t_min", 1),
            delta_t_max = mcfg.get("delta_t_max", 10),
            fc_hz       = mcfg.get("fc_hz",       5.18e9),
        )

    else:
        raise ValueError(f"Unknown method: {method}")


# ══════════════════════════════════════════════════════════════
#  STEP 4: Training & Evaluation
# ══════════════════════════════════════════════════════════════

def _step_train(cfg: dict, dirs: dict) -> dict:
    method = cfg["feature_extraction"]["method"]
    lcfg   = cfg.get("learning", {})
    mcfg   = cfg["feature_extraction"].get(method, {})

    os.makedirs(dirs["models"],    exist_ok=True)
    os.makedirs(dirs["res_learn"], exist_ok=True)

    _add_path(os.path.join(BASE_DIR, "src", "learning", "mlp"))

    # 공통 args namespace
    args = types.SimpleNamespace(
        feat_dir   = dirs["feat"],
        sanit_dir  = dirs["feat_input"],
        hidden     = lcfg.get("hidden",     [256, 128, 64]),
        activation = lcfg.get("activation", "relu"),
        epochs     = lcfg.get("epochs",      300),
        lr         = lcfg.get("lr",          1e-3),
        model_dir  = dirs["models"],
        result_dir = dirs["res_learn"],
        inference  = False,
        model_path = None,
        infer_file = None,
        infer_dir  = None,
        is_feat    = True,
    )

    if method == "dwt":
        import dwt_mlp as trainer
        args.wavelet = mcfg.get("wavelet", "sym3")
        args.level   = mcfg.get("level",   10)
        args.n_pca   = mcfg.get("n_pca",   6)
        X, y, subjects = trainer.load_dataset(
            args.feat_dir, args.sanit_dir,
            args.wavelet, args.level, args.n_pca,
        )

    elif method == "dfs":
        import dfs_mlp as trainer
        args.n_fft      = mcfg.get("n_fft",      64)
        args.hop        = mcfg.get("hop",          4)
        args.doppler_hz = mcfg.get("doppler_hz", 50.0)
        args.n_pca      = lcfg.get("dfs_n_pca",  32)
        X, y, subjects  = trainer.load_dataset(args)

    elif method == "sdp":
        import sdp_mlp as trainer
        args.n_lag = mcfg.get("n_lag", 20)
        args.wt    = mcfg.get("wt",   100)
        X, y, subjects = trainer.load_dataset(
            args.feat_dir, args.sanit_dir,
            args.n_lag, args.wt,
        )

    elif method == "tddfs":
        import tddfs_mlp as trainer
        args.delta_t_min = mcfg.get("delta_t_min", 1)
        args.delta_t_max = mcfg.get("delta_t_max", 10)
        args.fc_hz       = mcfg.get("fc_hz",       5.18e9)
        X, y, subjects   = trainer.load_dataset(args)

    else:
        raise ValueError(f"Unknown method: {method}")

    if len(X) == 0:
        raise RuntimeError(f"[Train] No valid samples from {args.feat_dir}")

    print(f"[Train] Loaded {len(X)} samples  feat_dim={X.shape[1]}")
    trainer.train_and_evaluate(X, y, subjects, args)

    # 저장된 meta JSON에서 지표 읽기
    meta_path = os.path.join(dirs["models"], f"{method}_mlp_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ══════════════════════════════════════════════════════════════
#  메인 파이프라인
# ══════════════════════════════════════════════════════════════

def run_pipeline(cfg: dict) -> dict:
    """
    YAML cfg에 따라 전체 파이프라인 실행.

    Returns
    -------
    metrics : dict  (test_accuracy, cv_mean, cv_std, ...)
    """
    dirs  = get_dirs(cfg)
    steps = cfg["experiment"].get("steps", ["extract", "train"])

    print(f"\n{'='*60}")
    print(f"  실험명 : {cfg['experiment']['name']}")
    print(f"  방법   : {cfg['feature_extraction']['method']}")
    sw = cfg.get("sliding_window", {})
    if sw.get("enabled", False):
        print(f"  윈도우 : {sw['window_sec']}s  hop={sw['hop_sec']}s")
    else:
        print(f"  윈도우 : 비활성화 (전체 신호)")
    print(f"  스텝   : {steps}")
    print(f"{'='*60}\n")

    metrics = {}

    if "preprocess" in steps:
        print("-- [Step 1] Preprocessing -----------------------------")
        asyncio.run(_step_preprocess(cfg, dirs))

    if "sanitize" in steps:
        print("-- [Step 2] Sanitization ------------------------------")
        asyncio.run(_step_sanitize(cfg, dirs))


    if "extract" in steps:
        print("-- [Step 3] Feature Extraction ------------------------")
        _step_extract(cfg, dirs)

    if "train" in steps:
        print("-- [Step 4] Training & Evaluation ---------------------")
        metrics = _step_train(cfg, dirs)

    return metrics


# ══════════════════════════════════════════════════════════════
#  유틸리티
# ══════════════════════════════════════════════════════════════

def _add_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)
