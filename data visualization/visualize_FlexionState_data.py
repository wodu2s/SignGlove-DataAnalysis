import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# --- 설정 상수 ---
# 분석할 고정 심볼 경로와 이름 설정
DEFAULT_SYMBOL_DIR = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data cleaning\backup\ㄱ_20250901_074541"
DEFAULT_SYMBOL = "ㄱ"

# flexion_state 폴더 목록
FLEXION_STATE_DIRS = ["1", "2", "3", "4", "5"]
# 통계 지표 순서
STATS_INDEX_ORDER = ['sum', 'mean', 'var', 'std', 'median', 'min', 'max', 'q1', 'q3']
# 결측치로 간주할 값 목록
NA_VALUES = ["", " ", "NA"]
# 기본 출력 경로
OUTPUT_ROOT = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data visualization\visualization"

# 한글 자모 판별용 집합
JAEUM_SET = set(list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"))
MOEUM_SET = set(list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"))

def classify_symbol(symbol: str) -> str:
    """심볼이 한글 자음/모음인지 판별하여 접미사 문자열을 반환합니다."""
    if not symbol:
        return "기타"
    ch = symbol[0]
    if ch in JAEUM_SET:
        return "자음"
    if ch in MOEUM_SET:
        return "모음"
    return "기타"

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
            'max': desc.get('max'), 'q1': desc.get('25%'), 'q3':
            desc.get('75%')
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

def create_and_save_visualizations_by_flexion_state(df, flex_columns, symbol, output_path):
    """
    각 flexion_state에 대해 'flex 1~5 라인을 같은 축'에 겹쳐 표시하는 그래프를 저장합니다.
    - x축: 각 flexion_state 내부의 샘플 인덱스(0..n-1)로 통일
    - y축: 센서 값 (600~1000 고정)
    - 파일명: FlexionState_{symbol}_FlexionState_{fs}.png
    """
    if 'flexion_state' not in df.columns:
        raise ValueError("데이터프레임에 'flexion_state' 컬럼이 없어 flexion_state별 시각화를 수행할 수 없습니다.")
    df = df.copy()
    df['sample_index'] = df.groupby('flexion_state').cumcount()

    symbol_kind = classify_symbol(symbol)

    for fs in FLEXION_STATE_DIRS:
        flexion_state_df = df[df['flexion_state'] == fs]
        if flexion_state_df.empty:
            continue

        plt.figure(figsize=(12, 7))
        plotted_any = False

        for col in flex_columns:
            if flexion_state_df[col].dropna().empty:
                continue
            
            plt.plot(flexion_state_df['sample_index'], flexion_state_df[col], label=f"{col}")
            plotted_any = True

        if not plotted_any:
            plt.close()
            continue

        plt.ylim(600, 1000)
        plt.grid(True)

        data_count = flexion_state_df[flex_columns].dropna(how='all').shape[0]
        spikes = int((flexion_state_df[flex_columns] >= 80).sum().sum())

        title = (
            f"Symbol: {symbol} | Flexion State: {fs}\n"
            f"Data Points (in this flexion_state): {data_count} | Spikes (>=80, in this flexion_state): {spikes}"
        )
        plt.title(title, fontsize=14)
        plt.xlabel('Sample Index (per Flexion State)')
        plt.ylabel('Flex (ADC)')
        plt.legend()

        img_filename = f"FlexionState_{symbol}_FlexionState_{fs}.png"
        save_path = os.path.join(output_path, img_filename)

        plt.savefig(save_path)
        print(f"  -> 통합 그래프 저장: '{save_path}'")
        plt.close()

def main():
    """메인 실행 함수"""
    print(f"심볼 '{DEFAULT_SYMBOL}'에 대한 통합 분석을 시작합니다.")
    print(f"기본 데이터 경로: {DEFAULT_SYMBOL_DIR}")

    all_flexion_state_dfs = []
    # 모든 flexion_state 폴더에서 데이터 로드 (+ flexion_state 컬럼 주입)
    for flexion_state in FLEXION_STATE_DIRS:
        flexion_state_data_path = os.path.join(DEFAULT_SYMBOL_DIR, flexion_state)
        if os.path.exists(flexion_state_data_path):
            print(f" -> flexion_state '{flexion_state}' 데이터 로드 중...")
            df = load_data_from_folder(flexion_state_data_path, NA_VALUES)
            if df is not None and not df.empty:
                df = df.copy()
                df['flexion_state'] = flexion_state  # flexion_state 식별자 주입
                all_flexion_state_dfs.append(df)
        else:
            print(f" -> 경고: flexion_state '{flexion_state}'의 데이터 경로를 찾을 수 없습니다: {flexion_state_data_path}")

    if not all_flexion_state_dfs:
        print("\n오류: 분석할 데이터가 없습니다.")
        return

    # 모든 flexion_state 데이터를 하나의 데이터프레임으로 결합
    combined_df = pd.concat(all_flexion_state_dfs, ignore_index=True)
    print("\n모든 flexion_state 데이터 결합 완료.")

    # flex 컬럼 식별
    flex_columns = sorted([col for col in combined_df.columns if col.startswith('flex')])
    if not flex_columns:
        print("오류: 'flex'로 시작하는 데이터 열을 찾을 수 없습니다.")
        return
        
    # flex 컬럼들을 숫자형으로 변환하고, 변환할 수 없는 값은 NaN으로 처리
    for col in flex_columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    # 출력 경로 설정 (e.g., .../visualization/ㄱ/Flexion_State/Flex_senser)
    symbol_output_path = os.path.join(OUTPUT_ROOT, DEFAULT_SYMBOL, "Flexion_State", "Flex_senser")
    os.makedirs(symbol_output_path, exist_ok=True)
    print(f"결과 저장 경로: {symbol_output_path}")

    # 통계 계산 및 저장 (요청에 따라 유지)
    print("\n통계 분석 중...")
    calculate_and_save_stats(combined_df, flex_columns, symbol_output_path)

    # 시각화 생성 및 저장: flex 센서 오버레이 버전
    print("\n시각화 생성 중...")
    create_and_save_visualizations_by_flexion_state(combined_df, flex_columns, DEFAULT_SYMBOL, symbol_output_path)

    print("\n분석이 완료되었습니다.")

if __name__ == '__main__':
    main()