# -*- coding: utf-8 -*-
"""
시계열 데이터 급격한 변화(Jump) 탐지 및 마스킹 도구

이 스크립트는 CSV 파일의 시계열 데이터에서 급격한 값의 변화(Jump)를 자동으로 탐지하고 마스킹합니다.
지정된 디렉토리에서 CSV 파일을 읽어, 채널별 임계값을 기반으로 급격한 변화 구간을 식별하고,
해당 구간을 NaN으로 마스킹한 후, 원본 데이터는 보존한 채 새로운 CSV 파일로 결과를 저장합니다.
"""
import os
import glob
import json
import logging
import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# --- 설정 상수 ---

# 입력 데이터 설정
# 중요: 분석할 데이터가 포함된 루트 디렉토리 경로를 지정하세요.
# 예시: r"C:\path\to\your\data\backup\ㄱ_20250901_074541"
DEFAULT_SYMBOL_DIR = r"C:\dev\SignGlove-DataAnalysis\unified_v2"
DEFAULT_SYMBOL = "ㄱ" # 분석할 글자 또는 클래스 (예: "ㄱ", "ㄴ")

# 출력 디렉토리 설정
# 중요: 정제된 데이터와 리포트가 저장될 루트 디렉토리입니다.
OUTPUT_ROOT = r"C:\dev\SignGlove-DataAnalysis\data cleaning\deleted_data_v2"

# --- 급격한 변화(Jump) 탐지 파라미터 ---

# Jump 탐지를 위한 임계값. 연속된 두 데이터 포인트 간의 절대값 차이가 이 값을 초과하면 Jump로 간주합니다.
THRESHOLDS = {
    "flex": 0.8,      # Flex 센서(flex1-flex5) 임계값
    "angle": 25.0     # 각도 센서(yaw, pitch, roll) 임계값
}

# 유효한 구간으로 간주될 최소 연속 데이터 포인트 수.
# 이 값보다 짧은 구간은 무시됩니다.
MIN_SEGMENT_LEN = 2

# 탐지된 두 Jump 구간 사이의 간격이 이 값(데이터 포인트 수)보다 작으면,
# 하나의 구간으로 병합됩니다.
MIN_GAP_TO_MERGE = 5

# --- 스크립트 설정 ---

# True로 설정하면 탐지된 모든 Jump 구간에 대한 상세 리포트를 저장합니다.
SAVE_SEGMENT_REPORTS = True

# Jump를 분석할 컬럼
FLEX_CHANNELS = [f"flex{i}" for i in range(1, 6)]
ANGLE_CHANNELS = ["yaw", "pitch", "roll"]
TARGET_CHANNELS = FLEX_CHANNELS + ANGLE_CHANNELS
TIMESTAMP_COL = "timestamp_ms"
REQUIRED_COLS = [TIMESTAMP_COL] + TARGET_CHANNELS

# --- 설정 종료 ---


def setup_logging(output_root: Path, run_timestamp: str):
    """콘솔과 파일에 모두 로깅하도록 설정합니다."""
    log_dir = output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{run_timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("로깅 설정 완료.")


def save_run_config(output_root: Path, run_timestamp: str):
    """현재 설정 상수들을 JSON 파일로 저장합니다."""
    config = {
        "run_timestamp": run_timestamp,
        "DEFAULT_SYMBOL_DIR": str(DEFAULT_SYMBOL_DIR),
        "DEFAULT_SYMBOL": DEFAULT_SYMBOL,
        "OUTPUT_ROOT": str(OUTPUT_ROOT),
        "THRESHOLDS": THRESHOLDS,
        "MIN_SEGMENT_LEN": MIN_SEGMENT_LEN,
        "MIN_GAP_TO_MERGE": MIN_GAP_TO_MERGE,
        "SAVE_SEGMENT_REPORTS": SAVE_SEGMENT_REPORTS,
        "TARGET_CHANNELS": TARGET_CHANNELS,
    }
    config_path = output_root / f"run_config_{run_timestamp}.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    logging.info(f"실행 설정이 {config_path}에 저장되었습니다.")


def load_and_prepare_data(file_path: Path) -> pd.DataFrame | None:
    """CSV를 로드하고, 타임스탬프를 파싱하며, 데이터를 정렬하고 유효성을 검사합니다."""
    try:
        df = pd.read_csv(file_path)
        
        if not all(col in df.columns for col in REQUIRED_COLS):
            logging.warning(f"{file_path} 파일을 건너뜁니다. 필수 컬럼이 없습니다.")
            return None

        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
        df = df.sort_values(by=TIMESTAMP_COL).drop_duplicates(subset=[TIMESTAMP_COL]).reset_index(drop=True)
        return df
    except Exception as e:
        logging.error(f"{file_path} 로드 또는 준비 실패: {e}")
        return None


def get_channel_type(channel: str) -> str:
    """채널이 'flex'인지 'angle'인지 결정합니다."""
    return "flex" if "flex" in channel else "angle"


def detect_jumps(df: pd.DataFrame) -> tuple[list, pd.DataFrame | None]:
    """
    데이터프레임의 모든 대상 채널에서 Jump 구간을 탐지합니다.
    
    반환값:
        튜플:
        - 리포트용 상세 정보가 담긴 딕셔너리 리스트.
        - 마스킹할 구간 정보가 담긴 데이터프레임 (컬럼: channel, start, end).
    """
    all_segments_info = []
    segments_to_mask = []

    for channel in TARGET_CHANNELS:
        channel_type = get_channel_type(channel)
        threshold = THRESHOLDS[channel_type]
        
        diff = df[channel].diff().abs()
        is_jump = diff > threshold
        
        if not is_jump.any():
            continue

        # Jump 시퀀스의 시작 및 종료 지점 찾기
        jump_indices = np.where(is_jump)[0]
        
        if len(jump_indices) == 0:
            continue

        # 연속된 인덱스를 세그먼트로 그룹화
        current_segment_start = jump_indices[0]
        for i in range(1, len(jump_indices)):
            if jump_indices[i] > jump_indices[i-1] + MIN_GAP_TO_MERGE:
                segments_to_mask.append({"channel": channel, "start": current_segment_start, "end": jump_indices[i-1]})
                current_segment_start = jump_indices[i]
        segments_to_mask.append({"channel": channel, "start": current_segment_start, "end": jump_indices[-1]})

    if not segments_to_mask:
        return [], None

    # 세그먼트 필터링 및 처리
    df_segments = pd.DataFrame(segments_to_mask)
    df_segments["n_points"] = df_segments["end"] - df_segments["start"] + 1
    df_segments = df_segments[df_segments["n_points"] >= MIN_SEGMENT_LEN].copy()

    if df_segments.empty:
        return [], None

    # 상세 리포트 채우기
    segment_id_counter = 0
    for _, seg in df_segments.iterrows():
        ch = seg["channel"]
        start_idx, end_idx = seg["start"], seg["end"]
        
        segment_data = df.loc[start_idx:end_idx]
        delta_data = df[ch].diff().abs().loc[start_idx:end_idx]

        all_segments_info.append({
            "segment_id": segment_id_counter,
            "channel": ch,
            "start_time": segment_data[TIMESTAMP_COL].iloc[0],
            "end_time": segment_data[TIMESTAMP_COL].iloc[-1],
            "peak_delta": delta_data.max(),
            "n_points": len(segment_data),
            "threshold_used": THRESHOLDS[get_channel_type(ch)],
            "method": "diff_threshold"
        })
        segment_id_counter += 1
        
    return all_segments_info, df_segments


def get_output_paths(input_path: Path, symbol: str, input_root: Path, output_root: Path) -> tuple[Path, Path]:
    """정제된 데이터와 리포트를 위한 출력 경로를 구성합니다."""
    try:
        relative_path = input_path.relative_to(input_root)
    except ValueError:
        # input_path가 input_root 내에 없는 경우를 정상적으로 처리합니다.
        # 경로 구조가 예상과 다를 때 발생할 수 있습니다.
        # 파일 이름과 일반적인 에피소드 이름을 사용합니다.
        relative_path = Path("episode_unknown") / input_path.name

    # 파일의 직속 부모 폴더를 에피소드로 간주합니다.
    episode = input_path.parent.name
    
    filename = input_path.name
    
    # 특정 파일을 위한 기본 출력 디렉토리
    base_output_dir = output_root / symbol / episode
    
    # 정제된 데이터 CSV 경로
    cleaned_csv_path = base_output_dir / filename
    
    # 구간 리포트 경로
    report_dir = base_output_dir / "_reports"
    report_csv_path = report_dir / f"{input_path.stem}_jump_segments.csv"
    
    # 디렉토리 생성
    base_output_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_SEGMENT_REPORTS:
        report_dir.mkdir(parents=True, exist_ok=True)
        
    return cleaned_csv_path, report_csv_path

def main():
    """메인 실행 함수"""
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root_path = Path(OUTPUT_ROOT)
    
    setup_logging(output_root_path, run_timestamp)
    save_run_config(output_root_path, run_timestamp)

    input_root_path = Path(DEFAULT_SYMBOL_DIR)
    if not input_root_path.is_dir():
        logging.error(f"입력 디렉토리를 찾을 수 없습니다: {input_root_path}")
        return

    logging.info(f"스캔 시작: {input_root_path}")
    
    # 모든 CSV 파일을 재귀적으로 찾기
    all_csv_files = list(glob.glob(os.path.join(input_root_path, "**", "*.csv"), recursive=True))
    
    if not all_csv_files:
        logging.warning("지정된 디렉토리에서 CSV 파일을 찾을 수 없습니다.")
        return

    # 파일 경로에 DEFAULT_SYMBOL이 포함된 파일만 필터링
    # 예: "...\unified\ㄱ\1\episode_..._ㄱ_1.csv"
    symbol_path_part = f"{os.sep}{DEFAULT_SYMBOL}{os.sep}"
    csv_files_to_process = [f for f in all_csv_files if symbol_path_part in f]

    if not csv_files_to_process:
        logging.warning(f"'{DEFAULT_SYMBOL}'에 해당하는 CSV 파일을 찾을 수 없습니다.")
        return

    logging.info(f"처리할 {len(csv_files_to_process)}개의 CSV 파일을 찾았습니다.")

    total_jumps_found = 0
    files_processed = 0

    for file_path_str in csv_files_to_process:
        file_path = Path(file_path_str)
        logging.info(f"--- 파일 처리 중: {file_path.name} ---")
        
        df = load_and_prepare_data(file_path)
        if df is None:
            continue

        segments_info, df_segments_to_mask = detect_jumps(df)

        if not segments_info:
            logging.info("Jump 구간을 찾지 못했습니다.")
            continue
        
        logging.info(f"모든 채널에서 {len(segments_info)}개의 Jump 구간을 찾았습니다.")
        total_jumps_found += len(segments_info)
        
        # 마스킹을 위한 복사본 생성
        df_cleaned = df.copy()
        
        for _, seg in df_segments_to_mask.iterrows():
            df_cleaned.loc[seg['start']:seg['end'], seg['channel']] = np.nan
            
        # 출력 경로 가져오기
        cleaned_path, report_path = get_output_paths(file_path, DEFAULT_SYMBOL, input_root_path, output_root_path)
        
        # 정제된 데이터 저장
        df_cleaned.to_csv(cleaned_path, index=False, encoding='utf-8-sig')
        logging.info(f"정제된 데이터를 다음 경로에 저장했습니다: {cleaned_path}")

        # 리포트 저장 (활성화된 경우)
        if SAVE_SEGMENT_REPORTS:
            report_df = pd.DataFrame(segments_info)
            report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
            logging.info(f"구간 리포트를 다음 경로에 저장했습니다: {report_path}")
            
        files_processed += 1

    logging.info("--- 처리 완료 ---")
    logging.info(f"Jump가 처리된 총 파일 수: {files_processed}")
    logging.info(f"탐지된 총 Jump 구간 수: {total_jumps_found}")


if __name__ == "__main__":
    main()