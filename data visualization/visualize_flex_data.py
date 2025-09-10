import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# --- 설정 상수 ---
# 분석할 고정 심볼 경로와 이름 설정
DEFAULT_SYMBOL_DIR = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data cleaning\backup\ㄱ_20250901_074541"
DEFAULT_SYMBOL = "ㄱ"

# 에피소드 폴더 목록
EPISODE_DIRS = ["1", "2", "3", "4", "5"]
# 통계 지표 순서
STATS_INDEX_ORDER = ['sum', 'mean', 'var', 'std', 'median', 'min', 'max', 'q1', 'q3']
# 결측치로 간주할 값 목록
NA_VALUES = ["", " ", "NA"]
# 기본 출력 경로
OUTPUT_ROOT = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data visualization\visualization"

def load_data_from_folder(folder_path, na_values):
    """폴더에서 모든 CSV 파일을 로드하고 단일 데이터프레임으로 병합합니다."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        return None
    
    df_list = [pd.read_csv(file, na_values=na_values) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    return df

def calculate_and_save_stats(df, flex_columns, output_path):
    """데이터프레임의 통계를 계산하고 CSV 파일로 저장합니다."""
    stats_list = []
    for col in flex_columns:
        if df[col].dropna().empty:
            continue
        
        desc = df[col].describe()
        stats = {
            'sum': df[col].sum(), 'mean': desc.get('mean'), 'var': df[col].var(),
            'std': desc.get('std'), 'median': df[col].median(), 'min': desc.get('min'),
            'max': desc.get('max'), 'q1': desc.get('25%'), 'q3': desc.get('75%')
        }
        stats_df = pd.DataFrame(stats, index=[col])
        stats_list.append(stats_df)

    if not stats_list:
        print("  -> 통계를 계산할 데이터가 없습니다.")
        return

    final_stats = pd.concat(stats_list).T
    final_stats = final_stats.reindex(STATS_INDEX_ORDER)
    
    stats_filename = os.path.join(output_path, "statistics_combined.csv")
    final_stats.to_csv(stats_filename)
    print(f"  -> 통합 통계 정보 저장: '{stats_filename}'")

def create_and_save_visualizations(df, flex_columns, symbol, output_path):
    """각 flex 센서에 대한 통합 시각화를 생성하고 이미지 파일로 저장합니다."""
    if 'sample_index' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'sample_index'})

    for col in flex_columns:
        if df[col].dropna().empty:
            continue

        plt.figure(figsize=(12, 7))
        plt.plot(df['sample_index'], df[col], label=col)
        plt.ylim(0, 1000)
        plt.grid(True)

        data_count = len(df[col].dropna())
        spikes = df[col][df[col] >= 80].count()
        
        title = f'Symbol: {symbol} (All Episodes) | Sensor: {col}\nData Points: {data_count} | Spikes (>=80): {spikes}'
        plt.title(title, fontsize=14)
        plt.xlabel('Sample Index (Combined)')
        plt.ylabel('Flex (ADC)')
        plt.legend()

        img_filename = f"combined_{symbol}_{col}.png"
        save_path = os.path.join(output_path, img_filename)
        plt.savefig(save_path)
        print(f"  -> 통합 그래프 저장: '{save_path}'")
        plt.close()

def main():
    """메인 실행 함수"""
    print(f"심볼 '{DEFAULT_SYMBOL}'에 대한 통합 분석을 시작합니다.")
    print(f"기본 데이터 경로: {DEFAULT_SYMBOL_DIR}")

    all_episode_dfs = []
    # 모든 에피소드 폴더에서 데이터 로드
    for episode in EPISODE_DIRS:
        episode_data_path = os.path.join(DEFAULT_SYMBOL_DIR, episode)
        if os.path.exists(episode_data_path):
            print(f" -> 에피소드 '{episode}' 데이터 로드 중...")
            df = load_data_from_folder(episode_data_path, NA_VALUES)
            if df is not None and not df.empty:
                all_episode_dfs.append(df)
        else:
            print(f" -> 경고: 에피소드 '{episode}'의 데이터 경로를 찾을 수 없습니다: {episode_data_path}")

    if not all_episode_dfs:
        print("\n오류: 분석할 데이터가 없습니다.")
        return

    # 모든 에피소드 데이터를 하나의 데이터프레임으로 결합
    combined_df = pd.concat(all_episode_dfs, ignore_index=True)
    print("\n모든 에피소드 데이터 결합 완료.")

    # flex 컬럼 식별
    flex_columns = sorted([col for col in combined_df.columns if col.startswith('flex')])
    if not flex_columns:
        print("오류: 'flex'로 시작하는 데이터 열을 찾을 수 없습니다.")
        return

    # 출력 경로 설정 (e.g., .../visualization/ㄱ/combined)
    symbol_output_path = os.path.join(OUTPUT_ROOT, DEFAULT_SYMBOL, "combined")
    os.makedirs(symbol_output_path, exist_ok=True)
    print(f"결과 저장 경로: {symbol_output_path}")

    # 통계 계산 및 저장
    print("\n통계 분석 중...")
    calculate_and_save_stats(combined_df, flex_columns, symbol_output_path)

    # 시각화 생성 및 저장
    print("\n시각화 생성 중...")
    create_and_save_visualizations(combined_df, flex_columns, DEFAULT_SYMBOL, symbol_output_path)

    print("\n분석이 완료되었습니다.")

if __name__ == '__main__':
    main()
