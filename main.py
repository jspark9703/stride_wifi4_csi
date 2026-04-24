"""
main.py
========
WiFi CSI Stride Classification — YAML 기반 단일 진입점

사용법
------
# 단일 실험 (기본값)
python main.py --config configs/default.yaml

# 특정 스텝만 실행
python main.py --config configs/default.yaml --steps extract train

# 즉석 파라미터 오버라이드
python main.py --config configs/default.yaml --set feature_extraction.dwt.level=8
python main.py --config configs/default.yaml --set sliding_window.enabled=true
python main.py --config configs/default.yaml --set learning.hidden=[512,256,128]

# Ablation Sweep (YAML 지정)
python main.py --ablation configs/ablation/window_size.yaml
python main.py --ablation configs/ablation/dwt_level.yaml
python main.py --ablation configs/ablation/full_sweep.yaml

# 방법 변경
python main.py --config configs/default.yaml --set feature_extraction.method=dfs

파이프라인 스텝
--------------
preprocess  → sanitize → [window] → extract → train

슬라이딩 윈도우
--------------
sliding_window.enabled=true 이면 sanitize 출력 → windowing → feature extraction
온라인 학습 대응: window_sec=0.8~1.0, hop_sec=0.1~0.5
"""

import argparse
import copy
import os
from datetime import datetime

from pipeline.config import load_config, set_nested, parse_value, validate_config
from pipeline.runner import run_pipeline
from pipeline import result_tracker as tracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ══════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="WiFi CSI Stride Classifier — YAML 기반 Ablation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행
  python main.py --config configs/default.yaml

  # 특징 추출 + 학습만
  python main.py --config configs/default.yaml --steps extract train

  # 파라미터 오버라이드
  python main.py --config configs/default.yaml --set feature_extraction.dwt.level=8

  # 슬라이딩 윈도우 활성화 (1.0초, 50% overlap)
  python main.py --config configs/default.yaml \\
    --set sliding_window.enabled=true \\
    --set sliding_window.window_sec=1.0 \\
    --set sliding_window.hop_sec=0.5

  # Ablation Sweep
  python main.py --ablation configs/ablation/window_size.yaml
  python main.py --ablation configs/ablation/dwt_level.yaml
        """,
    )

    # ── 모드 선택 (둘 중 하나만 사용)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config", metavar="PATH",
        help="단일 실험 YAML 설정 파일 경로",
    )
    group.add_argument(
        "--ablation", metavar="PATH",
        help="Ablation sweep YAML 파일 경로 (base_config + sweep 정의)",
    )

    # ── 단일 실험 옵션
    p.add_argument(
        "--steps", nargs="+",
        metavar="STEP",
        help="실행할 스텝만 지정 (preprocess sanitize window extract train)",
    )
    p.add_argument(
        "--set", action="append",
        metavar="KEY=VALUE",
        dest="overrides",
        help="dot-path 파라미터 오버라이드 (예: --set feature_extraction.dwt.level=8)",
    )

    return p


# ══════════════════════════════════════════════════════════════
#  단일 실험 실행
# ══════════════════════════════════════════════════════════════

def run_single(args) -> dict:
    cfg = load_config(args.config)

    # --steps 오버라이드
    if args.steps:
        cfg["experiment"]["steps"] = args.steps

    # --set 오버라이드
    if args.overrides:
        for kv in args.overrides:
            if "=" not in kv:
                print(f"[Warning] --set 형식 오류 (KEY=VALUE 필요): {kv}")
                continue
            key, _, val = kv.partition("=")
            set_nested(cfg, key.strip(), parse_value(val.strip()))

    validate_config(cfg)
    metrics = run_pipeline(cfg)

    if metrics:
        acc = metrics.get("test_accuracy", float("nan"))
        cv  = metrics.get("cv_mean",       float("nan"))
        std = metrics.get("cv_std",        float("nan"))
        print(f"\n[Result] Test Acc={acc:.4f}  CV={cv:.4f}±{std:.4f}")

    return metrics


# ══════════════════════════════════════════════════════════════
#  Ablation Sweep 실행
# ══════════════════════════════════════════════════════════════

def run_ablation(args) -> None:
    ablation_path = args.ablation
    abl_cfg = load_config(ablation_path)

    base_path = abl_cfg.get("base_config", "configs/default.yaml")
    if not os.path.isabs(base_path):
        base_path = os.path.join(BASE_DIR, base_path)

    base_cfg = load_config(base_path)

    # ablation 파일의 overrides 섹션 적용 (전역 오버라이드)
    global_overrides = abl_cfg.get("overrides", {})
    for dot_path, value in global_overrides.items():
        set_nested(base_cfg, dot_path, value)

    # CLI --set 오버라이드 (파일 override 보다 우선순위 높음)
    if args.overrides:
        for kv in args.overrides:
            if "=" in kv:
                key, _, val = kv.partition("=")
                set_nested(base_cfg, key.strip(), parse_value(val.strip()))

    sweep_list = abl_cfg.get("sweep", [])
    if not sweep_list:
        print("[Ablation] sweep 정의가 없습니다.")
        return

    # sweep 이름: ablation yaml 파일명 기반 + timestamp
    sweep_name = os.path.splitext(os.path.basename(ablation_path))[0]
    sweep_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(BASE_DIR, "results", f"{sweep_ts}_{sweep_name}")
    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, "ablation_summary.csv")

    # sweep 전체를 선형으로 전개 (각 param 독립 sweep)
    all_experiments = _expand_sweep(sweep_list, base_cfg)
    total = len(all_experiments)

    print(f"\n{'='*60}")
    print(f"  Ablation Sweep: {sweep_name}")
    print(f"  Timestamp     : {sweep_ts}")
    print(f"  총 실험 수    : {total}")
    print(f"  결과 디렉토리 : {result_dir}")
    print(f"{'='*60}")

    records = []

    for i, (exp_cfg, param_name, param_value) in enumerate(all_experiments, 1):
        val_str  = _value_to_str(param_value)
        exp_name = f"{param_name.split('.')[-1]}_{val_str}"
        exp_cfg["experiment"]["name"] = exp_name
        # sweep context 주입
        exp_cfg["experiment"]["_sweep_name"]    = sweep_name
        exp_cfg["experiment"]["_sweep_ts"]      = sweep_ts
        exp_cfg["experiment"]["_ablation_param"] = param_name

        print(f"\n[{i}/{total}] {param_name} = {param_value}")

        try:
            validate_config(exp_cfg)
            metrics = run_pipeline(exp_cfg)
        except Exception as e:
            print(f"  [ERROR] {e}")
            metrics = {"test_accuracy": float("nan"), "cv_mean": float("nan"), "cv_std": float("nan")}

        acc = float(metrics.get("test_accuracy", float("nan")))
        cv  = float(metrics.get("cv_mean",       float("nan")))
        std = float(metrics.get("cv_std",        float("nan")))
        dim = metrics.get("feature_dim",  "?")

        print(f"  → Test Acc={acc:.4f}  CV={cv:.4f}±{std:.4f}  dim={dim}")

        record = {
            "exp_name":      exp_name,
            "param":         param_name,
            "param_value":   val_str,
            "test_accuracy": f"{acc:.6f}",
            "cv_mean":       f"{cv:.6f}",
            "cv_std":        f"{std:.6f}",
            "feature_dim":   dim,
        }
        tracker.append_summary(summary_path, record)

        # 실험별 폴더에 config + metrics 저장 (runner가 경로 설정)
        run_dir = os.path.join(result_dir, "learning", exp_name)
        tracker.save_run(run_dir, exp_cfg, metrics)

        records.append({**record, "test_accuracy": acc, "cv_mean": cv, "cv_std": std})

    # 완료 후 비교 요약
    tracker.print_summary(records, sweep_name)

    # 단일 파라미터 sweep이면 그래프 생성
    unique_params = list({r["param"] for r in records})
    if len(unique_params) == 1:
        tracker.plot_ablation(summary_path, unique_params[0], result_dir)

    print(f"\n[Ablation] 완료 → {summary_path}")


# ══════════════════════════════════════════════════════════════
#  Sweep 전개 유틸리티
# ══════════════════════════════════════════════════════════════

def _expand_sweep(sweep_list: list, base_cfg: dict) -> list:
    """
    sweep 정의 → (cfg 복사본, param_name, param_value) 리스트 반환.
    각 param 그룹을 독립 sweep (나머지는 base 유지).
    """
    experiments = []
    for sweep_item in sweep_list:
        param_name = sweep_item["param"]
        values     = sweep_item["values"]
        for v in values:
            cfg_copy = copy.deepcopy(base_cfg)
            set_nested(cfg_copy, param_name, v)
            experiments.append((cfg_copy, param_name, v))
    return experiments


def _value_to_str(v) -> str:
    if isinstance(v, list):
        return "_".join(str(x) for x in v)
    if isinstance(v, float):
        return str(v).replace(".", "p")
    return str(v)


# ══════════════════════════════════════════════════════════════
#  진입점
# ══════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if args.ablation:
        # Ablation sweep 모드
        if not os.path.isabs(args.ablation):
            args.ablation = os.path.join(BASE_DIR, args.ablation)
        run_ablation(args)

    else:
        # 단일 실험 모드
        if not os.path.isabs(args.config):
            args.config = os.path.join(BASE_DIR, args.config)
        run_single(args)


if __name__ == "__main__":
    main()
