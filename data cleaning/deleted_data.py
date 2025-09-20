#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boxplot 기반 극단치 제거 파이프라인 (자/모 개별 심볼 폴더 단위)
- 플로우차트:
  시작 → 데이터 삽입 → IQR 경계 계산 → (범위 밖? 예: deleted_data.csv에 기록 후 원본에서 삭제 / 아니오: 유지) → csv 저장
- 작성자: ChatGPT
- 인코딩: utf-8-sig

핵심 정책
1) (symbol, class)별로 클래스 전체(≈1500개) 데이터를 메모리에서 합쳐 flex1..flex5에 대한 IQR 경계(Q1, Q3, IQR, lower, upper) 계산
2) 각 원본 CSV(약 300개)마다 경계를 적용하여 행 단위로 삭제 (어느 하나라도 flex1..flex5가 경계 밖이거나, NaN/비수치가 있으면 삭제)
3) 삭제된 행은 실행 단위 로그 파일: "deleted data_{symbol}.csv" (코드 파일이 있는 폴더에 생성, overwrite)
4) 원본 CSV는 백업 후 덮어쓰기 (기본). --no-backup 옵션으로 끌 수 있음.

주의
- 행 인덱스(row_index_before_delete)는 CSV를 읽었을 때의 0-based index로 기록(헤더 제외).
"""

import argparse
import sys
import os
import shutil
import glob
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# =========================
# 설정 상수
# =========================

CLASS_MAP = {
    1: "Fully Extended",
    2: "Slightly Extended",
    3: "Normal",
    4: "Slightly Bent",
    5: "Fully Bent",
}

DEFAULT_SYMBOL_DIR = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\unified\ㅣ"
DEFAULT_SYMBOL = "ㅣ"

# Outlier 판정 대상 컬럼
FLEX_COLS = ["flex1", "flex2", "flex3", "flex4", "flex5"]

# IQR 계수 (박스플롯 관례)
IQR_K = 1.5

# 입출력 설정
DEFAULT_ENCODING = "utf-8-sig"
DEFAULT_SEP = ","


# =========================
# 유틸
# =========================

def list_class_csvs(symbol_dir: str) -> Dict[int, List[str]]:
    """
    심볼 폴더(예: unified/ㄱ/) 아래 1~5 클래스 폴더의 CSV 경로를 딕셔너리로 반환.
    ex) {1: [...csvs], 2: [...], ..., 5: [...]}
    """
    class_to_files = {}
    for cls in range(1, 6):
        cls_dir = os.path.join(symbol_dir, str(cls))
        if not os.path.isdir(cls_dir):
            print(f"[경고] 클래스 폴더가 없습니다: {cls_dir}", file=sys.stderr)
            class_to_files[cls] = []
            continue
        files = sorted(glob.glob(os.path.join(cls_dir, "*.csv")))
        class_to_files[cls] = files
    return class_to_files


def read_csv_safe(path: str,
                  encoding: str = DEFAULT_ENCODING,
                  sep: str = DEFAULT_SEP) -> pd.DataFrame:
    """
    CSV를 안전하게 읽어오되, FLEX_COLS가 없으면 오류를 던짐.
    """
    try:
        df = pd.read_csv(path, encoding=encoding, sep=sep)
    except Exception as e:
        raise RuntimeError(f"CSV 읽기 실패: {path} ({e})")

    # 필수 컬럼 체크
    missing = [c for c in FLEX_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락 {missing} in {path}")

    return df


def to_numeric_or_nan(series: pd.Series) -> pd.Series:
    """
    비수치 → NaN으로 강제 변환
    """
    return pd.to_numeric(series, errors="coerce")


def compute_bounds_for_class(all_class_dfs: List[pd.DataFrame]) -> Dict[str, Tuple[float, float, float, float, float]]:
    """
    클래스 내 모든 파일(≈1500행)을 메모리에서 합쳐 FLEX_COLS별 박스플롯 경계를 계산.
    반환: {col: (Q1, Q3, IQR, lower, upper)}
    """
    if not all_class_dfs:
        raise ValueError("경계 계산용 데이터프레임이 비어 있습니다.")

    concat_df = pd.concat(all_class_dfs, ignore_index=True)

    # 비수치 → NaN
    for col in FLEX_COLS:
        concat_df[col] = to_numeric_or_nan(concat_df[col])

    # 전체가 NaN인 컬럼 방어
    bounds = {}
    for col in FLEX_COLS:
        col_series = concat_df[col].dropna()
        if col_series.empty:
            # 전부 NaN이면 경계를 만들 수 없음 → 넓은 경계로 처리
            bounds[col] = (np.nan, np.nan, np.nan, -np.inf, np.inf)
            continue

        q1 = np.nanpercentile(col_series, 25)
        q3 = np.nanpercentile(col_series, 75)
        iqr = q3 - q1
        lower = q1 - IQR_K * iqr
        upper = q3 + IQR_K * iqr
        bounds[col] = (q1, q3, iqr, lower, upper)

    return bounds


def build_reason_row(row: pd.Series, bounds: Dict[str, Tuple[float, float, float, float, float]]) -> List[str]:
    """
    해당 행이 outlier인 경우, 어떤 컬럼이 어떤 방식으로 경계를 벗어났는지 이유를 리스트로 반환.
    또한 NaN/비수치(=coerce 후 NaN)도 'is NaN'으로 이유를 남김.
    """
    reasons = []
    for col in FLEX_COLS:
        val = row[col]
        # NaN -> 비수치/결측
        if pd.isna(val):
            reasons.append(f"{col} is NaN")
            continue

        _, _, _, lower, upper = bounds[col]
        if val < lower:
            reasons.append(f"{col} < lower")
        elif val > upper:
            reasons.append(f"{col} > upper")
    return reasons


def backup_file(src_path: str, backup_root: str) -> str:
    """
    src_path를 backup_root/상대경로 로 복사. 상위 디렉터리 생성 보장.
    반환: 백업된 경로
    """
    rel = os.path.relpath(src_path, start=backup_root)  # 임시 rel 계산 방지용
    # 올바른 백업 경로 구성: backup_root/<원본 드라이브 루트 이후 경로>가 아니라,
    # backup_root/<원본의 상대 경로를 'symbol_dir 기준'으로 유지>가 바람직.
    # 여기서는 호출부에서 backup_dest를 구성하도록 함. (이 함수는 단순 파일 복제)
    dest_path = backup_root
    # 상위 폴더 생성
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy2(src_path, dest_path)
    return dest_path


def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# =========================
# 메인 처리
# =========================

def process_symbol(symbol_dir: str,
                   symbol: str,
                   dry_run: bool = False,
                   do_backup: bool = True,
                   encoding: str = DEFAULT_ENCODING,
                   sep: str = DEFAULT_SEP):
    """
    하나의 심볼(예: ㄱ) 폴더를 처리.
    - 클래스별로 IQR 경계 계산 (전체 1500행 기반)
    - 각 파일별로 outlier/NaN/비수치 행 제거
    - 삭제 로그를 코드 파일 폴더에 "deleted data_{symbol}.csv"로 저장 (실행마다 새로 생성)
    - 원본은 백업 후 덮어쓰기
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(script_dir, f"deleted data_{symbol}.csv")

    # 로그 헤더 준비 (overwrite)
    log_cols = ["timestamp", "class", "source_file", "row_index_before_delete"] + FLEX_COLS + ["reason"]
    log_df = pd.DataFrame(columns=log_cols)

    class_to_files = list_class_csvs(symbol_dir)

    # 백업 루트
    backup_root = None
    if do_backup and not dry_run:
        backup_root = os.path.join(script_dir, "backup", datetime.now().strftime("%Y%m%d_%H%M%S"))

    # 각 클래스별 경계 계산을 위해 먼저 전체 로딩
    class_bounds: Dict[int, Dict[str, Tuple[float, float, float, float, float]]] = {}

    for cls, files in class_to_files.items():
        if not files:
            print(f"[정보] 클래스 {cls}: 처리할 CSV가 없습니다.")
            continue

        all_dfs = []
        for f in files:
            df = read_csv_safe(f, encoding=encoding, sep=sep)
            # FLEX만 수치 변환
            for col in FLEX_COLS:
                df[col] = to_numeric_or_nan(df[col])
            all_dfs.append(df)

        bounds = compute_bounds_for_class(all_dfs)
        class_bounds[cls] = bounds

        # 진행 로그(요약)
        print(f"[경계] 클래스 {cls} ({CLASS_MAP.get(cls, '')})")
        for col in FLEX_COLS:
            q1, q3, iqr, lower, upper = bounds[col]
            print(f"  - {col}: Q1={q1:.4f} Q3={q3:.4f} IQR={iqr:.4f} lower={lower:.4f} upper={upper:.4f}")

    # 각 파일에 경계 적용 → 삭제 → 저장
    for cls, files in class_to_files.items():
        if not files or cls not in class_bounds:
            continue
        bounds = class_bounds[cls]

        for src in files:
            df = read_csv_safe(src, encoding=encoding, sep=sep)

            # 1) 판정용 숫자 변환 프레임(num)만 따로 만든다. (파일 저장에는 절대 사용하지 않음)
            num = df.copy()
            for col in FLEX_COLS:
                num[col] = to_numeric_or_nan(num[col])

            # 2) 행별 사유 수집(판정은 num로, 기록 값은 df 원본 값 사용)
            reasons_per_row: List[List[str]] = []
            for _, row in num.iterrows():
                reasons = []
                for col in FLEX_COLS:
                    val = row[col]
                    if pd.isna(val):
                        reasons.append(f"{col} is NaN")
                        continue
                    _, _, _, lower, upper = bounds[col]
                    if val < lower:
                        reasons.append(f"{col} < lower")
                    elif val > upper:
                        reasons.append(f"{col} > upper")
                reasons_per_row.append(reasons)

            # 3) 삭제 마스크 (num로 계산) - 어떤 flex라도 NaN 또는 경계 밖이면 삭제
            to_delete_mask = num.apply(
                lambda r: (
                    any(pd.isna(r[c]) for c in FLEX_COLS) or
                    any(
                        (r[c] < bounds[c][3]) or (r[c] > bounds[c][4])
                        for c in FLEX_COLS
                        if not pd.isna(r[c])
                    )
                ),
                axis=1
            )

            # 4) 삭제 로그는 df(원본 값)로 기록
            if to_delete_mask.any():
                del_idx = np.where(to_delete_mask.values)[0]  # 0-based
                for idx in del_idx:
                    row_orig = df.iloc[idx]   # 원본 값 그대로
                    reasons = reasons_per_row[idx] or ["unknown"]
                    log_row = {
                        "timestamp": timestamp_run,
                        "class": cls,
                        "source_file": os.path.relpath(src, start=symbol_dir),
                        "row_index_before_delete": int(idx),
                        **{col: row_orig[col] for col in FLEX_COLS},  # 원본 값 기록
                        "reason": "; ".join(reasons),
                    }
                    log_df.loc[len(log_df)] = log_row

            # 5) 실제 저장도 df(원본)에서 행만 제거
            cleaned = df[~to_delete_mask].reset_index(drop=True)

            if dry_run:
                print(f"[DryRun] {src}: 삭제 {int(to_delete_mask.sum())}행, 유지 {len(cleaned)}행")
            else:
                if do_backup and backup_root:
                    rel_from_symbol = os.path.relpath(src, start=symbol_dir)
                    backup_dest = os.path.join(backup_root, rel_from_symbol)
                    ensure_parent_dir(backup_dest)
                    shutil.copy2(src, backup_dest)

                cleaned.to_csv(src, index=False, encoding=encoding)
                print(f"[저장] {src}: 삭제 {int(to_delete_mask.sum())}행, 유지 {len(cleaned)}행 (덮어쓰기 완료)")
                
            # 실제 삭제 적용
            # 실제 삭제 적용 (원본 df에서 행만 제거)
            cleaned = df[~to_delete_mask].reset_index(drop=True)


            # 저장: 원본은 그대로 두고, 같은 폴더에 새 파일로 저장
            # episode_20250819_191331_ㄴ_1.csv  ->  cleaning_20250819_191331_ㄴ_1.csv
            base = os.path.basename(src)
            name, ext = os.path.splitext(base)
            parts = name.split("_")

            if len(parts) >= 5 and parts[0].lower() == "episode":
                # ['episode','YYYYMMDD','HHMMSS','심볼','클래스']
                target = "_".join(parts[1:]) + ext               # '20250819_191331_ㄴ_1.csv'
            else:
                # 예외: 포맷이 다르면 전체 이름 그대로 뒤에 붙이기
                target = name + ext

            output_file = f"cleaning_{target}"                   # 'cleaning_20250819_191331_ㄴ_1.csv'
            output_path = os.path.join(os.path.dirname(src), output_file)

            if dry_run:
                print(f"[DryRun] {src} -> {output_path}: 삭제 {int(to_delete_mask.sum())}행, 유지 {len(cleaned)}행")
            else:
                # (선택) 원본 백업은 유지하고 싶으면 남겨둠
                if do_backup and backup_root:
                    rel_from_symbol = os.path.relpath(src, start=symbol_dir)
                    backup_dest = os.path.join(backup_root, rel_from_symbol)
                    ensure_parent_dir(backup_dest)
                    shutil.copy2(src, backup_dest)

                # 새 파일로 저장
                cleaned.to_csv(output_path, index=False, encoding=encoding)
                print(f"[저장] {output_path}: 삭제 {int(to_delete_mask.sum())}행, 유지 {len(cleaned)}행 (새 파일 저장)")

            
    # 삭제 로그 저장
    if dry_run:
        print(f"[DryRun] 삭제 로그 미생성. 생성 대상 경로: {log_path}")
        # 미리보기(상위 5행)
        print(log_df.head(5).to_string(index=False))
    else:
        # 코드 파일이 있는 폴더에 생성 (overwrite)
        log_df.to_csv(log_path, index=False, encoding=encoding)
        print(f"[로그] 삭제 로그 저장: {log_path} (총 {len(log_df)}행)")

    print("[완료] 처리 종료")


# =========================
# 엔트리포인트
# =========================

def main():
    parser = argparse.ArgumentParser(description="Boxplot(IQR) 기반 극단치 제거 (심볼 폴더 단위)")
    parser.add_argument("--symbol-dir", type=str, default=DEFAULT_SYMBOL_DIR,  # ✅ 경로 기본값
                    help="심볼 폴더 경로 (예: '.../unified/ㄱ') (미지정 시 DEFAULT_SYMBOL_DIR 사용)")
    parser.add_argument(
    "--symbol",
    type=str,
    default=DEFAULT_SYMBOL,       # ← 심볼명 기본값
    help="심볼명(예: 'ㄱ') — 로그 파일명에 사용됨: deleted data_{symbol}.csv"
)

    parser.add_argument("--dry-run", action="store_true",
                        help="실제 파일 덮어쓰지 않고 동작만 시뮬레이션")
    parser.add_argument("--no-backup", action="store_true",
                        help="백업을 생성하지 않음")
    parser.add_argument("--encoding", type=str, default=DEFAULT_ENCODING,
                        help=f"입출력 인코딩 (기본: {DEFAULT_ENCODING})")
    parser.add_argument("--sep", type=str, default=DEFAULT_SEP,
                        help=f"CSV 구분자 (기본: '{DEFAULT_SEP}')")
    args = parser.parse_args()

     # ✅ 여기서 디버그 찍기
    print("[DEBUG] CWD:", os.getcwd())
    print("[DEBUG] script_dir:", os.path.dirname(os.path.abspath(__file__)))
    print("[DEBUG] symbol_dir 인자 값:", args.symbol_dir)

    symbol_dir = args.symbol_dir
    symbol = args.symbol
    dry_run = args.dry_run
    do_backup = not args.no_backup
    encoding = args.encoding
    sep = args.sep

    if not os.path.isdir(symbol_dir):
        print(f"[오류] 심볼 폴더가 존재하지 않습니다: {symbol_dir}", file=sys.stderr)
        sys.exit(1)

    # 클래스 폴더(1~5) 존재 간단 체크
    missing_cls = [str(c) for c in range(1, 6) if not os.path.isdir(os.path.join(symbol_dir, str(c)))]
    if missing_cls:
        print(f"[경고] 다음 클래스 폴더가 없습니다: {', '.join(missing_cls)}", file=sys.stderr)

    process_symbol(symbol_dir, symbol, dry_run=dry_run, do_backup=do_backup, encoding=encoding, sep=sep)


if __name__ == "__main__":
    main()