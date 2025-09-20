import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# --- 설정 상수 ---
DEFAULT_SYMBOL_DIR = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\unified\ㅣ"
DEFAULT_SYMBOL = "ㅣ"
EPISODE_DIRS = ["1", "2", "3", "4", "5"]
STATS_INDEX_ORDER = ['sum', 'mean', 'var', 'std', 'median', 'min', 'max', 'q1', 'q3']
NA_VALUES = ["", " ", "NA"]
OUTPUT_ROOT = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data visualization\visualization"

# --- 함수 정의 ---

def load_data_from_folder(folder_path, na_values):
    """폴더에서 모든 CSV 파일을 로드하고 데이터프레임 리스트로 반환합니다."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        return []
    
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, na_values=na_values)
            df.attrs['filename'] = os.path.basename(file)
            df_list.append(df)
        except Exception as e:
            print(f"  - 파일 읽기 오류 '{os.path.basename(file)}': {e}")
    return df_list

def calculate_and_save_stats(df, flex_columns, output_path):
    """(기존 함수) 데이터프레임의 통계를 계산하고 CSV 파일로 저장합니다."""
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

def create_and_save_combined_visualizations(df, flex_columns, symbol, output_path):
    """(기존 함수명 변경) 모든 데이터를 합친 통합 시각화를 생성합니다."""
    if 'sample_index' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'sample_index'})
    for col in flex_columns:
        if df[col].dropna().empty:
            continue
        plt.figure(figsize=(12, 7))
        plt.plot(df['sample_index'], df[col], label=col)
        plt.ylim(600, 1000)
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

def create_and_save_individual_visualizations(df, flex_columns, symbol, output_path):
    """(신규 함수) 개별 CSV 파일에 대한 시각화를 생성합니다."""
    original_filename = df.attrs.get('filename', 'unknown_file')
    plt.figure(figsize=(12, 7))
    for col in flex_columns:
        plt.plot(df.index, df[col], label=col)
    plt.ylim(600, 1000)
    plt.grid(True)
    plt.title(f'Symbol: {symbol} | File: {original_filename}', fontsize=14)
    plt.xlabel('Sample Index')
    plt.ylabel('Flex (ADC)')
    plt.legend()
    base_name, _ = os.path.splitext(original_filename)
    img_filename = f"{base_name}.png"
    save_path = os.path.join(output_path, img_filename)
    plt.savefig(save_path)
    print(f"  -> 개별 그래프 저장: '{save_path}'")
    plt.close()

def main():
    """메인 실행 함수"""
    print(f"심볼 '{DEFAULT_SYMBOL}'에 대한 분석을 시작합니다.")
    print(f"기본 데이터 경로: {DEFAULT_SYMBOL_DIR}")

    # --- 출력 폴더 설정 ---
    combined_output_path = os.path.join(OUTPUT_ROOT, DEFAULT_SYMBOL, "combined")
    individual_base_output_path = os.path.join(OUTPUT_ROOT, DEFAULT_SYMBOL, "individual")
    os.makedirs(combined_output_path, exist_ok=True)
    os.makedirs(individual_base_output_path, exist_ok=True)
    print(f"통합 결과 저장 경로: {combined_output_path}")
    print(f"개별 결과 기본 경로: {individual_base_output_path}")

    all_dfs = []
    # --- 데이터 로드 및 개별 시각화 ---
    print("\n데이터 로드 및 개별 시각화 생성 중...")
    for episode in EPISODE_DIRS:
        episode_data_path = os.path.join(DEFAULT_SYMBOL_DIR, episode)
        if os.path.exists(episode_data_path):
            print(f" -> 에피소드 '{episode}' 데이터 로드 중...")
            
            # --- 수정: 에피소드별 개별 출력 폴더 생성 ---
            individual_episode_output_path = os.path.join(individual_base_output_path, episode)
            os.makedirs(individual_episode_output_path, exist_ok=True)

            dfs_from_folder = load_data_from_folder(episode_data_path, NA_VALUES)
            
            for df in dfs_from_folder:
                flex_columns = sorted([col for col in df.columns if col.startswith('flex')])
                if flex_columns:
                    # (수정) 개별 파일 시각화 함수에 에피소드별 경로 전달
                    create_and_save_individual_visualizations(df, flex_columns, DEFAULT_SYMBOL, individual_episode_output_path)
                all_dfs.append(df)
        else:
            print(f" -> 경고: 에피소드 '{episode}'의 데이터 경로를 찾을 수 없습니다.")

    if not all_dfs:
        print("\n오류: 분석할 데이터가 없습니다.")
        return

    # --- 통합 데이터 분석 (기존 로직) ---
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print("\n모든 에피소드 데이터 결합 완료.")

    flex_columns = sorted([col for col in combined_df.columns if col.startswith('flex')])
    if not flex_columns:
        print("오류: 'flex'로 시작하는 데이터 열을 찾을 수 없습니다.")
        return

    print("\n통합 통계 분석 중...")
    calculate_and_save_stats(combined_df, flex_columns, combined_output_path)

    print("\n통합 시각화 생성 중...")
    create_and_save_combined_visualizations(combined_df, flex_columns, DEFAULT_SYMBOL, combined_output_path)

    print("\n모든 분석이 완료되었습니다.")

if __name__ == '__main__':
    main()
