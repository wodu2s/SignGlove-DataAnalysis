# -*- coding: utf-8 -*-
"""
저장 경로: C:/Users/jjy06/OneDrive/바탕 화면/DeepBot/AIoT 프로젝트/Data Analysis/data visualization/ypl_state_plot.py
목적: 각 센서(yaw/pitch/roll)에 대해 Ep1~Ep5(flexion_state 1~5)를 동일한 축에 겹쳐서 시각화합니다.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import logging

# ---------------------- 상수 ----------------------

# !!! 중요: 이 경로를 실제 데이터 디렉토리로 수정해주세요 !!!
# 이 경로는 심볼 하위 디렉토리(예: 'ㄱ', 'ㄴ', 'ㄷ')를 포함하는 루트 폴더입니다.
DEFAULT_SYMBOL_DIR = "C:/Users/jjy06/OneDrive/바탕 화면/DeepBot/AIoT 프로젝트/unified"  # <-- 경로를 수정해주세요

DEFAULT_SYMBOL = "ㄱ"
FLEXION_STATE_DIRS = ["1", "2", "3", "4", "5"]
STATS_INDEX_ORDER = ['sum', 'mean', 'var', 'std', 'median', 'min', 'max', 'q1', 'q3']
NA_VALUES = ["", " ", "NA"]

# 출력 루트 (심볼을 동적으로 포함)
OUTPUT_ROOT = f"C:/Users/jjy06/OneDrive/바탕 화면/DeepBot/AIoT 프로젝트/Data Analysis/data visualization/visualization/{DEFAULT_SYMBOL}/state_ypl_plot"

# 플롯 옵션
Y_RANGE = (-180, 180)
FIGSIZE = (12, 7)
PLOT_DPI = 150

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------- 보조 함수 ----------------------

def load_data_from_folder(folder_path: str, na_values: list) -> pd.DataFrame | None:
    """폴더의 모든 CSV를 단일 DataFrame으로 로드합니다."""
    if not os.path.isdir(folder_path):
        logging.warning(f"데이터 폴더를 찾을 수 없습니다: {folder_path}")
        return None
        
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        logging.warning(f"폴더에서 CSV 파일을 찾을 수 없습니다: {folder_path}")
        return None
        
    frames = []
    for f in csv_files:
        try:
            frames.append(pd.read_csv(f, na_values=na_values))
        except Exception as e:
            logging.error(f"파일 읽기 오류로 건너뜁니다: {f} ({e})")
            
    if not frames:
        return None
        
    return pd.concat(frames, ignore_index=True)

def detect_ypl_columns(df: pd.DataFrame) -> list:
    """
    yaw, pitch, roll 컬럼 이름을 특정 순서로 탐지하여 반환합니다.
    검색은 대소문자를 구분하지 않으며, 접두사 기반으로 예비 검색을 수행합니다.
    """
    if df is None or df.empty:
        return []

    cols_lower = {c.lower(): c for c in df.columns}
    ordered_ypl = []

    for base in ["yaw", "pitch", "roll"]:
        found_col = None
        if base in cols_lower:
            found_col = cols_lower[base]
        else:
            # 접두사 예비 검색
            for col_name in df.columns:
                if col_name.lower().startswith(base):
                    found_col = col_name
                    break
        if found_col:
            ordered_ypl.append(found_col)
            
    return ordered_ypl

def compute_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """주어진 컬럼에 대한 기술 통계를 계산합니다."""
    stats_list = []
    for col in cols:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if series.empty:
            continue
            
        desc = series.describe()
        stats = {
            'sum': series.sum(),
            'mean': desc.get('mean'),
            'var': series.var(),
            'std': desc.get('std'),
            'median': series.median(),
            'min': desc.get('min'),
            'max': desc.get('max'),
            'q1': desc.get('25%'),
            'q3': desc.get('75%'),
        }
        stats_list.append(pd.DataFrame(stats, index=[col]))
        
    if not stats_list:
        return pd.DataFrame()
        
    out_df = pd.concat(stats_list).T
    return out_df.reindex(STATS_INDEX_ORDER)

# ---------------------- 플로팅 함수 ----------------------

def plot_overlay_per_sensor(df: pd.DataFrame, sensor_col: str, out_dir: str):
    """
    각 센서에 대해 하나의 플롯을 생성하고, 모든 에피소드(flexion states)의 데이터를 겹쳐서 그립니다.
    """
    if 'flexion_state' not in df.columns:
        logging.error("플로팅 실패: 'flexion_state' 컬럼이 없습니다.")
        return

    df = df.copy()
    df['sample_index'] = df.groupby('flexion_state').cumcount()

    plt.figure(figsize=FIGSIZE)
    plotted_something = False
    total_points = 0

    for i, state in enumerate(FLEXION_STATE_DIRS, start=1):
        subset = df[df['flexion_state'] == state]
        if subset.empty or subset[sensor_col].dropna().empty:
            continue
            
        y_values = pd.to_numeric(subset[sensor_col], errors='coerce')
        x_values = subset['sample_index']
        
        plt.plot(x_values, y_values, label=f"Ep{i}")
        total_points += y_values.dropna().shape[0]
        plotted_something = True

    if not plotted_something:
        plt.close()
        logging.warning(f"센서 '{sensor_col}'에 대한 데이터가 없어 플롯을 건너뜁니다.")
        return

    plt.ylim(*Y_RANGE)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlabel("Sample Index (per Flexion State)")
    plt.ylabel("Angle (deg)")
    plt.legend(title="Episode")

    title = f"Symbol: {DEFAULT_SYMBOL} (Episodes 1-5 Overlay) | Sensor: {sensor_col}"
    subtitle = f"Points (all): {total_points}"
    plt.title(f"{title}\n{subtitle}", fontsize=12)

    filename = f"ypl_{DEFAULT_SYMBOL}_{sensor_col}_EpisodesOverlay.png"
    save_path = os.path.join(out_dir, filename)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=PLOT_DPI)
    plt.close()
    logging.info(f"플롯 저장 완료: {save_path}")

# ---------------------- 메인 실행 ----------------------

def main():
    """데이터 분석 및 시각화를 총괄하는 메인 함수."""
    logging.info(f"--- 심볼 '{DEFAULT_SYMBOL}'에 대한 ypl 오버레이 분석 시작 ---")
    
    # 출력 디렉토리 존재 확인
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    logging.info(f"출력물 저장 경로: {OUTPUT_ROOT}")

    # 모든 flexion state 폴더에서 데이터 로드 및 결합
    all_frames = []
    symbol_dir = os.path.join(DEFAULT_SYMBOL_DIR, DEFAULT_SYMBOL)
    logging.info(f"데이터 소스 경로: {symbol_dir}")

    for state in FLEXION_STATE_DIRS:
        path = os.path.join(symbol_dir, state)
        df = load_data_from_folder(path, NA_VALUES)
        if df is not None and not df.empty:
            df['flexion_state'] = state
            all_frames.append(df)

    if not all_frames:
        logging.error("분석 중단: 로드할 데이터가 없습니다.")
        return

    combined_df = pd.concat(all_frames, ignore_index=True)
    logging.info("모든 상태의 데이터를 성공적으로 결합했습니다.")

    # YPL 컬럼 탐지 및 숫자형으로 변환
    ypl_columns = detect_ypl_columns(combined_df)
    if len(ypl_columns) < 3:
        logging.error(f"분석 중단: 모든 YPL 컬럼을 찾을 수 없습니다. 탐지된 컬럼: {ypl_columns}")
        return
    logging.info(f"탐지된 YPL 컬럼: {ypl_columns}")
    
    for col in ypl_columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    # 통계 계산 및 저장
    stats_df = compute_stats(combined_df, ypl_columns)
    if not stats_df.empty:
        stats_path = os.path.join(OUTPUT_ROOT, "statistics_combined.csv")
        stats_df.to_csv(stats_path)
        logging.info(f"통계 저장 완료: {stats_path}")
    else:
        logging.warning("유효한 숫자 데이터가 없어 통계를 생성하지 않았습니다.")

    # 각 센서에 대해 모든 에피소드를 겹쳐서 플롯 생성
    for sensor_col in ypl_columns:
        plot_overlay_per_sensor(combined_df, sensor_col, OUTPUT_ROOT)

    logging.info("--- 분석이 성공적으로 완료되었습니다. ---")

if __name__ == "__main__":
    main()