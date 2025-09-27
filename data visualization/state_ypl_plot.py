import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import logging

# --- 설정 상수 ---

# !!! 중요: 이 경로를 실제 데이터 디렉토리로 설정해주세요 !!!
# 이 경로는 심볼 하위 디렉토리('ㄱ', 'ㄴ', 'ㄷ' 등)를 포함하는 루트 폴더입니다.
DEFAULT_SYMBOL_DIR = "C:/Users/jjy06/OneDrive/바탕 화면/DeepBot/AIoT 프로젝트/unified" # <-- 경로를 수정해주세요

# 별도로 지정하지 않을 경우 처리할 기본 심볼.
DEFAULT_SYMBOL = "ㄴ"

# 각 flexion_state에 대한 하위 디렉토리.
FLEXION_STATE_DIRS = ["1", "2", "3", "4", "5"]

# 통계 CSV 파일의 인덱스 순서.
STATS_INDEX_ORDER = ['sum', 'mean', 'var', 'std', 'median', 'min', 'max', 'q1', 'q3']

# CSV 로드 시 결측치(NA)로 처리할 값 목록.
NA_VALUES = ["", " ", "NA", "N/A", "NaN"]

# 출력 파일을 저장할 루트 디렉토리 (변경 가능).
OUTPUT_ROOT = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data visualization\visualization"

# 시각화 설정.
Y_LIM_ORIENTATION = (-180, 180)
SPIKE_THRESHOLD = 20
PLOT_DPI = 200

# 기본 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 보조 함수 ---

def classify_symbol(symbol: str) -> str:
    """
    하나의 한글 문자를 '자음', '모음', '기타'로 분류합니다.
    더 고급 심볼 처리에 사용할 수 있는 보조 함수입니다.
    """
    if not symbol or len(symbol) > 1:
        return 'other'
    char_code = ord(symbol)
    # 한글 음절 (여기서는 사용되지 않지만, 완전성을 위해 포함)
    if 0xAC00 <= char_code <= 0xD7A3:
        return 'syllable'
    # 한글 자모 (자음 및 모음)
    if 0x3131 <= char_code <= 0x314E:
        return 'consonant'
    if 0x314F <= char_code <= 0x3163:
        return 'vowel'
    return 'other'


# --- 핵심 함수 ---

def load_data_from_folder(folder_path: str, na_values: list) -> pd.DataFrame:
    """
    주어진 폴더의 모든 CSV 파일을 단일 pandas DataFrame으로 로드합니다.
    
    Args:
        folder_path (str): CSV 파일이 들어있는 폴더 경로.
        na_values (list): NA/NaN으로 인식할 문자열 리스트.

    Returns:
        pd.DataFrame: 폴더의 모든 CSV가 병합된 DataFrame.
                      폴더가 없거나 CSV 파일이 없으면 빈 DataFrame을 반환합니다.
    """
    if not os.path.isdir(folder_path):
        logging.warning(f"데이터 폴더를 찾을 수 없습니다: {folder_path}")
        return pd.DataFrame()
    
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        logging.warning(f"폴더에서 CSV 파일을 찾을 수 없습니다: {folder_path}")
        return pd.DataFrame()

    df_list = [pd.read_csv(f, na_values=na_values) for f in csv_files]
    return pd.concat(df_list, ignore_index=True)


def get_orientation_columns(df: pd.DataFrame) -> list:
    """
    DataFrame에서 yaw, pitch, roll 컬럼을 자동으로 식별합니다.
    검색은 대소문자를 구분하지 않으며 다양한 이름 변형을 허용합니다.
    
    Args:
        df (pd.DataFrame): 입력 DataFrame.

    Returns:
        list: 식별된 yaw, pitch, roll 컬럼 이름 리스트.
    
    Raises:
        ValueError: 방향 컬럼('yaw', 'pitch', 'roll')을 찾을 수 없는 경우.
    """
    orientation_keywords = ['yaw', 'pitch', 'roll']
    found_columns = {key: None for key in orientation_keywords}
    
    for col in df.columns:
        col_lower = col.lower()
        for keyword in orientation_keywords:
            if keyword in col_lower and not found_columns[keyword]:
                found_columns[keyword] = col
    
    # 컬럼을 찾지 못한 키워드는 제외
    result_columns = [col for col in found_columns.values() if col]
    
    if not result_columns:
        raise ValueError("DataFrame에서 방향 컬럼(yaw, pitch, roll)을 찾을 수 없습니다.")
        
    logging.info(f"식별된 방향 컬럼: {result_columns}")
    return result_columns


def calculate_and_save_stats(df: pd.DataFrame, ori_columns: list, output_path: str):
    """
    주어진 방향 컬럼에 대한 기술 통계를 계산하고 CSV 파일로 저장합니다.
    
    Args:
        df (pd.DataFrame): 데이터가 포함된 DataFrame.
        ori_columns (list): 분석할 방향 컬럼 이름 리스트.
        output_path (str): 통계 파일을 저장할 디렉토리 경로.
    """
    stats_df = df[ori_columns].agg(['sum', 'mean', 'var', 'std', 'median', 'min', 'max'])
    q1 = df[ori_columns].quantile(0.25).rename('q1')
    q3 = df[ori_columns].quantile(0.75).rename('q3')
    
    stats_df = pd.concat([stats_df, q1.to_frame().T, q3.to_frame().T])
    stats_df = stats_df.reindex(STATS_INDEX_ORDER)
    
    stats_filepath = os.path.join(output_path, "statistics_combined.csv")
    stats_df.to_csv(stats_filepath)
    logging.info(f"병합된 통계 정보를 저장했습니다: {stats_filepath}")


def create_and_save_visualizations_by_episode_orientation(df: pd.DataFrame, ori_columns: list, symbol: str, output_path: str):
    """
    Generates and saves plots for each episode (flexion_state), showing orientation data over time.
    
    Args:
        df (pd.DataFrame): The complete DataFrame with all episodes.
        ori_columns (list): The list of orientation column names to plot.
        symbol (str): The symbol being processed (for titles).
        output_path (str): The directory path to save the plot images.
    """
    # Use groupby to generate an intra-episode index
    df['episode_index'] = df.groupby('flexion_state').cumcount()

    for episode in sorted(df['flexion_state'].unique()):
        episode_df = df[df['flexion_state'] == episode].copy()
        
        if episode_df.empty:
            logging.warning(f"Skipping visualization for Episode {episode}: No data available.")
            continue

        plt.figure(figsize=(12, 6))
        
        total_spikes = 0
        for col in ori_columns:
            # Ensure data is numeric and calculate spikes
            episode_df[col] = pd.to_numeric(episode_df[col], errors='coerce')
            spikes = episode_df[col].diff().abs() >= SPIKE_THRESHOLD
            total_spikes += spikes.sum()
            
            plt.plot(episode_df['episode_index'], episode_df[col], label=f'{col} (Spikes: {spikes.sum()})')

        data_points = len(episode_df)
        title = (f"Symbol: {symbol} | Episode: {episode} | Orientation Analysis\n"
                 f"Data Points: {data_points} | Total Spikes (diff >= {SPIKE_THRESHOLD}): {total_spikes}")
        
        plt.title(title)
        plt.xlabel("Sample Index within Episode")
        plt.ylabel("Orientation (degrees)")
        plt.ylim(Y_LIM_ORIENTATION)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plot_filename = f"visualized_FlexionState_ep{episode}_orientation.png"
        plot_filepath = os.path.join(output_path, plot_filename)
        plt.savefig(plot_filepath, dpi=PLOT_DPI)
        plt.close()
        
        logging.info(f"Saved orientation plot for Episode {episode} to: {plot_filepath}")


# --- 메인 실행 흐름 ---

def main():
    """
    데이터 처리 및 시각화 파이프라인을 실행하는 메인 함수.
    """
    logging.info("--- 데이터 분석 및 시각화 스크립트 시작 ---")
    
    # 1. 각 flexion_state에 대한 데이터 로드
    all_data = []
    symbol_path = os.path.join(DEFAULT_SYMBOL_DIR, DEFAULT_SYMBOL)
    
    for state in FLEXION_STATE_DIRS:
        folder_path = os.path.join(symbol_path, state)
        logging.info(f"데이터 로드 중: {folder_path}")
        
        df_state = load_data_from_folder(folder_path, NA_VALUES)
        
        if not df_state.empty:
            df_state['flexion_state'] = int(state)
            all_data.append(df_state)
            logging.info(f"flexion_state {state}에 대해 {len(df_state)}개의 행을 로드했습니다.")
        else:
            logging.warning(f"flexion_state {state}에 대한 데이터를 로드하지 못했습니다. 건너뜁니다.")

    if not all_data:
        logging.error("어떤 flexion_state에 대해서도 데이터를 로드할 수 없습니다. 스크립트를 종료합니다.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    logging.info(f"모든 flexion_state의 데이터를 성공적으로 병합했습니다. 총 행 수: {len(combined_df)}")

    # 2. 방향 컬럼 식별 및 숫자형으로 변환
    try:
        orientation_columns = get_orientation_columns(combined_df)
        for col in orientation_columns:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        # 변환 후 방향 데이터가 누락된 행 삭제
        initial_rows = len(combined_df)
        combined_df.dropna(subset=orientation_columns, inplace=True)
        if len(combined_df) < initial_rows:
            logging.warning(f"숫자가 아닌 방향 데이터를 포함한 {initial_rows - len(combined_df)}개의 행을 삭제했습니다.")

    except ValueError as e:
        logging.error(e)
        logging.error("방향 컬럼 없이는 진행할 수 없습니다. 스크립트를 종료합니다.")
        return

    # 3. 출력 디렉토리 생성
    output_dir = os.path.join(OUTPUT_ROOT, DEFAULT_SYMBOL, "ypl_state_plot")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"출력 디렉토리를 생성했거나 이미 존재합니다: {output_dir}")

    # 4. 통계 계산 및 저장
    calculate_and_save_stats(combined_df, orientation_columns, output_dir)

    # 5. 시각화 생성 및 저장
    create_and_save_visualizations_by_episode_orientation(combined_df, orientation_columns, DEFAULT_SYMBOL, output_dir)

    logging.info("--- 스크립트가 성공적으로 완료되었습니다! ---")


if __name__ == "__main__":
    main()