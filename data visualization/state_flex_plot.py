import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# --- 설정 상수 ---
DEFAULT_SYMBOL_DIR = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data cleaning\backup\ㄷ_20250901_080751"
DEFAULT_SYMBOL = "ㄷ"

FLEXION_STATE_DIRS = ["1", "2", "3", "4", "5"]
STATS_INDEX_ORDER = ['sum', 'mean', 'var', 'std', 'median', 'min', 'max', 'q1', 'q3']
NA_VALUES = ["", " ", "NA"]
OUTPUT_ROOT = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data visualization\visualization"

# 시각화 / 스파이크 파라미터
Y_MIN, Y_MAX = 600, 1000
SPIKE_THRESHOLD = 80  # |diff| >= 80 이면 스파이크

# (선택) 한글 자모 판별 – 필요 시 사용
JAEUM_SET = set(list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"))
MOEUM_SET = set(list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"))

def classify_symbol(symbol: str) -> str:
    if not symbol:
        return "기타"
    ch = symbol[0]
    if ch in JAEUM_SET:
        return "자음"
    if ch in MOEUM_SET:
        return "모음"
    return "기타"

def load_data_from_folder(folder_path, na_values):
    """폴더의 모든 CSV를 로드해 하나의 DF로 결합."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        return None
    df_list = [pd.read_csv(file, na_values=na_values) for file in csv_files]
    return pd.concat(df_list, ignore_index=True)

def calculate_and_save_stats(df, flex_columns, output_path):
    """flex* 컬럼 통계를 저장."""
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

# ---------------------- 새로 추가/수정된 부분 ----------------------

def get_flex_sensor_columns(df):
    """
    flex 센서 컬럼만 정확히 추출.
    - 'flexion_state'는 제외 (기존 버그 방지).
    - 우선적으로 flex1~flex5를 사용. 없으면 'flex*' 중 앞에서 5개.
    """
    all_flex = [c for c in df.columns if c.lower().startswith('flex')]
    all_flex = [c for c in all_flex if c.lower() not in ('flexion_state', 'flexionstate')]

    preferred = [f'flex{i}' for i in range(1, 6) if f'flex{i}' in df.columns]
    if len(preferred) == 5:
        return preferred

    # 부족하면 나머지 flex*에서 채워 5개까지만 사용
    rest = [c for c in all_flex if c not in preferred]
    picked = preferred + rest
    if not picked:
        raise ValueError("유효한 flex 센서 컬럼이 없습니다.")
    return picked[:5]

def create_and_save_visualizations_by_episode_sensors(df, flex_columns, symbol, output_path):
    """
    ✅ 요구사항: '각 에피소드별로' flex 센서 5개를 한 그래프에 겹쳐 그림 → 총 5개 이미지 생성.
    - x축: 해당 에피소드 내부 샘플 인덱스 (0..n-1)
    - y축: 각 flex 센서 값
    - 제목: Symbol/ Episode / Sensor + Data Points / Spikes(절대차분 기준) 요약
    - 저장 파일명: visualized_FlexionState_ep{ep}_flex_sensors.png
    """
    if 'flexion_state' not in df.columns:
        raise ValueError("'flexion_state' 컬럼이 없어 에피소드별 시각화를 할 수 없습니다.")

    # 에피소드 내부 인덱스
    work = df.copy()
    work['sample_index'] = work.groupby('flexion_state').cumcount()

    # 에피소드 루프
    for ep in FLEXION_STATE_DIRS:
        ep_df = work[work['flexion_state'] == ep].copy()
        if ep_df.empty:
            print(f"  -> 경고: 에피소드 {ep} 데이터 없음. 스킵.")
            continue

        # 선택된 flex 컬럼만 사용 (숫자화는 main에서 처리)
        used_cols = [c for c in flex_columns if c in ep_df.columns and ep_df[c].notna().any()]
        if not used_cols:
            print(f"  -> 경고: 에피소드 {ep}에서 사용 가능한 flex 데이터 없음. 스킵.")
            continue

        plt.figure(figsize=(12, 7))
        for col in used_cols:
            plt.plot(ep_df['sample_index'], ep_df[col], linewidth=1.2, label=col)

        # 스파이크: 각 센서의 절대차분 기준 합산
        spikes_total = 0
        for col in used_cols:
            diffs = ep_df[col].diff().abs()
            spikes_total += int((diffs >= SPIKE_THRESHOLD).sum())

        # 유효 데이터 포인트 수(행 기준)
        data_points = int(ep_df[used_cols].notna().any(axis=1).sum())

        plt.ylim(Y_MIN, Y_MAX)
        plt.grid(True, alpha=0.3)
        title = (
            f"Symbol: {symbol} | Episode: {ep} | Sensor: flex\n"
            f"Data Points (Ep{ep}): {data_points:,} | Spikes (>= {SPIKE_THRESHOLD}, Ep{ep}): {spikes_total:,}"
        )
        plt.title(title, fontsize=13)
        plt.xlabel("Sample Index (per Episode)")
        plt.ylabel("Flex (ADC)")
        plt.legend(ncol=len(used_cols))
        plt.tight_layout()

        img_filename = f"visualized_FlexionState_ep{ep}_flex_sensors.png"
        save_path = os.path.join(output_path, img_filename)
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"  -> 에피소드 {ep} 그래프 저장: '{save_path}'")

# ---------------------- 여기까지 ----------------------

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

    # ✅ flex 센서 컬럼 정확히 추출( flexion_state 제외 )
    flex_columns = get_flex_sensor_columns(combined_df)

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

    # ✅ 에피소드별(1~5)로 flex 센서 5개를 한 그래프에 – 총 5개 이미지 생성
    print("\n시각화 생성 중 (에피소드별 센서 오버레이)...")
    create_and_save_visualizations_by_episode_sensors(combined_df, flex_columns, DEFAULT_SYMBOL, symbol_output_path)

    print("\n분석이 완료되었습니다.")

if __name__ == '__main__':
    main()
