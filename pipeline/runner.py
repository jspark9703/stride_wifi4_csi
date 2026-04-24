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

BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASELINE_ROOT = os.path.join(BASE_DIR, "data", "ablation", "baseline")


# ══════════════════════════════════════════════════════════════
#  디렉토리 구성 유틸리티
# ══════════════════════════════════════════════════════════════

def _pick(affected: set, step: str, fresh_path: str, reuse_path: str) -> str:
    """step이 affected에 포함되면 fresh_path, 아니면 reuse_path(baseline) 반환."""
    return fresh_path if step in affected else reuse_path


def _affected_steps(param: str) -> set:
    """
    ablation 파라미터 dot-path의 최상위 섹션으로
    '새로 실행해야 하는 스텝' 집합을 반환.

    | param prefix        | 새로 실행할 스텝                              |
    |---------------------|-----------------------------------------------|
    | preprocessing.*     | preprocess, sanitize, extract, train          |
    | sanitization.*      | sanitize, extract, train                      |
    | sliding_window.*    | window, preprocess, sanitize, extract, train  |
    | feature_extraction.*| extract, train                                |
    | learning.*          | train                                         |
    """
    prefix = param.split(".")[0] if param else ""
    return {
        "preprocessing":      {"preprocess", "sanitize", "extract", "train"},
        "sanitization":       {"sanitize",   "extract",  "train"},
        "sliding_window":     {"window", "preprocess", "sanitize", "extract", "train"},
        "feature_extraction": {"extract",    "train"},
        "learning":           {"train"},
    }.get(prefix, set())


# ══════════════════════════════════════════════════════════════
#  디렉토리 구성
# ══════════════════════════════════════════════════════════════

def get_dirs(cfg: dict) -> dict:
    """
    실험 모드별 데이터/결과 경로 딕셔너리 반환.

    모드 분기
    ---------
    1. baseline 단일 실행  (exp_name == "baseline", sweep 없음)
       → data/ablation/baseline/{preprocessed,sanitization,windowed,feature_extraction}
    2. 일반 단일 실행  (그 외 exp_name, sweep 없음)
       → 기존 분산 구조 유지 (data/sanitization/, data/feature_extraction/)
    3. Ablation sweep  (_sweep_name, _sweep_ts 존재)
       → _ablation_param 로 영향 스텝 계산,
         영향 스텝은 sweep 경로, 나머지는 baseline 재사용
    """
    date_tag       = cfg["experiment"].get("date_tag", "")
    exp_name       = cfg["experiment"]["name"]
    method         = cfg["feature_extraction"]["method"]
    sweep_name     = cfg["experiment"].get("_sweep_name")
    sweep_ts       = cfg["experiment"].get("_sweep_ts")
    ablation_param = cfg["experiment"].get("_ablation_param", "")
    sw_enabled     = cfg.get("sliding_window", {}).get("enabled", False)

    # ── 모드 1: Ablation sweep ─────────────────────────────────
    if sweep_name and sweep_ts:
        affected  = _affected_steps(ablation_param)
        sweep_exp = os.path.join(
            BASE_DIR, "data", "ablation",
            f"{sweep_ts}_{sweep_name}", exp_name
        )
        res_root  = os.path.join(BASE_DIR, "results", f"{sweep_ts}_{sweep_name}")

        prep     = _pick(affected, "preprocess",
                         os.path.join(sweep_exp, "preprocessed"),
                         os.path.join(BASELINE_ROOT, "preprocessed"))
        sanit    = _pick(affected, "sanitize",
                         os.path.join(sweep_exp, "sanitization"),
                         os.path.join(BASELINE_ROOT, "sanitization"))
        windowed = _pick(affected, "window",
                         os.path.join(sweep_exp, "windowed"),
                         os.path.join(BASELINE_ROOT, "windowed"))
        feat     = _pick(affected, "extract",
                         os.path.join(sweep_exp, "feature_extraction"),
                         os.path.join(BASELINE_ROOT, "feature_extraction", method))

        res_learn = os.path.join(res_root, "learning", exp_name)
        res_feat  = os.path.join(res_root, "feature_extraction", exp_name)
        models    = os.path.join(res_root, "learning", exp_name, "models")

    # ── 모드 2: Baseline 단일 실행 ────────────────────────────
    elif exp_name == "baseline":
        prep     = os.path.join(BASELINE_ROOT, "preprocessed")
        sanit    = os.path.join(BASELINE_ROOT, "sanitization")
        windowed = os.path.join(BASELINE_ROOT, "windowed")
        feat     = os.path.join(BASELINE_ROOT, "feature_extraction", method)

        res_root  = os.path.join(BASE_DIR, "results", "baseline")
        res_learn = os.path.join(res_root, "learning")
        res_feat  = os.path.join(res_root, "feature_extraction")
        models    = os.path.join(res_root, "learning", "models")

    # ── 모드 3: 일반 단일 실행 (기존 분산 구조) ───────────────
    else:
        prep     = os.path.join(BASE_DIR, "data", "sanitization", "preprocessed", date_tag)
        sanit    = os.path.join(BASE_DIR, "data", "sanitization", "sanitization", date_tag)
        windowed = os.path.join(BASE_DIR, "data", "windowed", date_tag)
        feat     = os.path.join(BASE_DIR, "data", "feature_extraction", method, exp_name)

        res_root  = os.path.join(BASE_DIR, "results", exp_name)
        res_learn = os.path.join(res_root, "learning")
        res_feat  = os.path.join(res_root, "feature_extraction")
        models    = os.path.join(res_root, "learning", "models")

    # feature extraction 입력은 항상 sanitize 결과
    # (window → preprocess → sanitize → extract 순서이므로)

    return {
        "raw":        os.path.join(BASE_DIR, "data", "raw", date_tag),
        "prep":       prep,
        "sanit":      sanit,
        "windowed":   windowed,
        "feat":       feat,
        "feat_input": sanit,
        "json":       os.path.join(BASE_DIR, "data", "logs", "sanitization", "json"),
        "plots":      os.path.join(BASE_DIR, "data", "logs", "sanitization", "plots"),
        "log_win":    os.path.join(BASE_DIR, "data", "logs", "windowing"),
        "res_feat":   res_feat,
        "res_learn":  res_learn,
        "models":     models,
    }


# ══════════════════════════════════════════════════════════════
#  STEP 0 & 1: Preprocessing
# ══════════════════════════════════════════════════════════════

async def _step_preprocess(cfg: dict, dirs: dict) -> None:
    """
    [Step 1] Preprocessing: CSV → preprocessed NPZ.

    sliding_window.enabled=True  → windowed CSV (dirs["windowed"]) 를 입력으로 사용.
    sliding_window.enabled=False → raw CSV (dirs["raw"]) 를 입력으로 사용.
    windowing 자체는 window 스텝에서 완료되므로 window_sec=0.
    """
    pcfg       = cfg.get("preprocessing", {})
    sw_enabled = cfg.get("sliding_window", {}).get("enabled", False)

    _add_path(os.path.join(BASE_DIR, "src", "sanitization", "scripts"))
    import preprocess

    # windowing 이 완료된 CSV 또는 raw CSV 를 입력으로
    input_dir = dirs["windowed"] if sw_enabled else dirs["raw"]

    await preprocess.run_preprocessing(
        raw_dir          = input_dir,
        out_dir          = dirs["prep"],
        json_dir         = dirs["json"],
        target_fs        = pcfg.get("target_fs",        100),
        max_gap_ms       = pcfg.get("max_gap_ms",        20.0),
        hampel_enabled   = pcfg.get("hampel_enabled",   True),
        hampel_window    = pcfg.get("hampel_window",    7),
        hampel_threshold = pcfg.get("hampel_threshold", 5.0),
        lowpass_enabled  = pcfg.get("lowpass_enabled",  False),
        lowpass_cutoff   = pcfg.get("lowpass_cutoff",   11.0),
        window_sec       = 0,   # windowing 은 window 스텝에서 완료
        hop_sec          = 0,
        run_id           = cfg["experiment"].get("run_id"),
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
    """
    [Step W] Window: raw CSV → windowed CSV.
    파이프라인에서 preprocess 스텝 바로 앞에 자동 실행됨.
    raw/ 의 CSV 파일들을 시간 기반 슬라이딩 윈도우로 분할 → windowed/ 에 저장.
    """
    from pipeline.windowing import apply_sliding_window_csv

    sw = cfg["sliding_window"]

    apply_sliding_window_csv(
        raw_dir    = dirs["raw"],
        out_dir    = dirs["windowed"],
        window_sec = sw.get("window_sec", 2.5),
        hop_sec    = sw.get("hop_sec",    0.5),
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

    파이프라인 순서
    --------------
    raw/ → [window] → windowed/ → [preprocess] → preprocessed/
         → [sanitize] → sanitization/ → [extract] → [train]

    window 스텝은 sliding_window.enabled=True 이고 steps에 preprocess 가 포함될 때
    자동으로 preprocess 전에 삽입됨.

    Returns
    -------
    metrics : dict  (test_accuracy, cv_mean, cv_std, ...)
    """
    dirs  = get_dirs(cfg)
    steps = cfg["experiment"].get("steps", ["extract", "train"])
    sw    = cfg.get("sliding_window", {})
    sw_enabled = sw.get("enabled", False)

    print(f"\n{'='*60}")
    print(f"  실험명 : {cfg['experiment']['name']}")
    print(f"  방법   : {cfg['feature_extraction']['method']}")
    if sw_enabled:
        print(f"  윈도우 : {sw['window_sec']}s  hop={sw['hop_sec']}s")
    else:
        print(f"  윈도우 : 비활성화 (전체 신호)")
    print(f"  스텝   : {steps}")
    print(f"{'='*60}\n")

    metrics = {}

    # ── [Step W] Window (raw CSV → windowed CSV) ────────────────
    # sliding_window 활성화 + preprocess 스텝 포함 시 자동 삽입
    if sw_enabled and "preprocess" in steps:
        print("-- [Step W] Window Segmentation (CSV) -----------------")
        _step_window(cfg, dirs)

    # ── [Step 1] Preprocess (windowed or raw CSV → NPZ) ─────────
    if "preprocess" in steps:
        print("-- [Step 1] Preprocessing -----------------------------")
        asyncio.run(_step_preprocess(cfg, dirs))

    # ── [Step 2] Sanitize (preprocessed NPZ → sanitized NPZ) ────
    if "sanitize" in steps:
        print("-- [Step 2] Sanitization ------------------------------")
        asyncio.run(_step_sanitize(cfg, dirs))

    # ── [Step 3] Extract (sanitized NPZ → features) ─────────────
    if "extract" in steps:
        print("-- [Step 3] Feature Extraction ------------------------")
        _step_extract(cfg, dirs)

    # ── [Step 4] Train ───────────────────────────────────────────
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
