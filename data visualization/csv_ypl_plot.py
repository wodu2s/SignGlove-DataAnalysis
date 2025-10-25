import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# --- 설정 상수 ---
DEFAULT_SYMBOL_DIR = r"C:\dev\SignGlove-DataAnalysis\unified_hb\ㅇ"
DEFAULT_SYMBOL = "ㅇ"
EPISODE_DIRS = ["1", "2", "3", "4", "5"]
NA_VALUES = ["", " ", "NA"]
OUTPUT_ROOT = r"C:\dev\SignGlove-DataAnalysis\data visualization\visualization"

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

def create_visualization(df, title, output_path, filename):
    """데이터프레임의 yaw, pitch, roll 데이터를 시각화하고 저장합니다."""
    data_columns = sorted([col for col in df.columns if col in ['yaw', 'pitch', 'roll']])
    if not data_columns:
        print(f"  -> '{title}'에 'yaw', 'pitch', 'roll' 데이터가 없습니다.")
        return

    plt.figure(figsize=(12, 7))
    if 'sample_index' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'sample_index'})

    for col in data_columns:
        plt.plot(df['sample_index'], df[col], label=col)

    plt.ylim(-80, 80)
    plt.grid(True)
    plt.title(title, fontsize=14)
    plt.xlabel('Sample Index (Combined)')
    plt.ylabel('Sensor Value')
    plt.legend()
    
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path)
    print(f"  -> 통합 그래프 저장: '{save_path}'")
    plt.close()

def create_single_column_visualization(df, column, title, output_path, filename):
    """데이터프레임의 단일 컬럼을 시각화하고 저장합니다."""
    if column not in df.columns:
        print(f"  -> '{title}'에 '{column}' 데이터가 없습니다.")
        return

    plt.figure(figsize=(12, 7))
    if 'sample_index' not in df.columns:
        df = df.reset_index().rename(columns={'index': 'sample_index'})

    plt.plot(df['sample_index'], df[column], label=column)

    plt.ylim(-80, 80)
    plt.grid(True)
    plt.title(title, fontsize=14)
    plt.xlabel('Sample Index (Combined)')
    plt.ylabel('Sensor Value')
    plt.legend()
    
    save_path = os.path.join(output_path, filename)
    plt.savefig(save_path)
    print(f"  -> 통합 그래프 저장: '{save_path}'")
    plt.close()

def main():
    """메인 실행 함수"""
    print(f"심볼 '{DEFAULT_SYMBOL}'에 대한 분석을 시작합니다.")
    print(f"기본 데이터 경로: {DEFAULT_SYMBOL_DIR}")

    base_output_path = os.path.join(OUTPUT_ROOT, DEFAULT_SYMBOL, "ypl")

    # --- 에피소드별 데이터 처리 ---
    all_dfs = []
    for episode in EPISODE_DIRS:
        episode_input_path = os.path.join(DEFAULT_SYMBOL_DIR, episode)
        episode_output_path = os.path.join(base_output_path, episode)
        os.makedirs(episode_output_path, exist_ok=True)

        print(f"\n에피소드 '{episode}' 처리 중...")
        
        dfs_from_folder = load_data_from_folder(episode_input_path, NA_VALUES)
        if not dfs_from_folder:
            print(f" -> 에피소드 '{episode}'에 데이터가 없습니다.")
            continue
        
        all_dfs.extend(dfs_from_folder)
        
        for df in dfs_from_folder:
            original_filename = df.attrs.get('filename', 'unknown_file')
            base_name, _ = os.path.splitext(original_filename)
            
            title = f'Symbol: {DEFAULT_SYMBOL} | Episode: {episode} | File: {original_filename}'
            filename = f"{base_name}.png"
            create_visualization(df, title, episode_output_path, filename)

    # --- 전체 데이터 처리 ---
    if not all_dfs:
        print("\n오류: 분석할 데이터가 없습니다.")
        return

    combined_output_path = os.path.join(base_output_path, "combined")
    os.makedirs(combined_output_path, exist_ok=True)
    
    print("\n전체 데이터 처리 중...")
    
    total_combined_df = pd.concat(all_dfs, ignore_index=True)
    
    for col in ['yaw', 'pitch', 'roll']:
        title = f'Symbol: {DEFAULT_SYMBOL} | All Episodes | {col.capitalize()}'
        filename = f"combined_{col}.png"
        create_single_column_visualization(total_combined_df, col, title, combined_output_path, filename)

    print("\n모든 분석이 완료되었습니다.")

if __name__ == '__main__':
    main()