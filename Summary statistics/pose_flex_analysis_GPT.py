import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 설정 ---
UNIFIED_DIR = r"C:\dev\SignGlove-DataAnalysis\unified_v2"
FLEX_COLS = [f'flex{i}' for i in range(1, 6)]
ROT_COLS = ['yaw', 'pitch', 'roll']
REQUIRED_COLS = FLEX_COLS + ROT_COLS

# --- 1단계: 데이터 로딩 및 기본 RII 계산 ---

def load_data(path: str) -> pd.DataFrame:
    """지정된 경로 하위의 모든 CSV 파일을 로드하여 단일 데이터프레임으로 반환합니다."""
    print(f"데이터 로드 시작: {path}")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"입력 디렉터리를 찾을 수 없습니다: {path}")

    df_list = []
    symbols_in_path = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    for symbol in symbols_in_path:
        symbol_path = os.path.join(path, symbol)
        files = glob.glob(os.path.join(symbol_path, "**", "*.csv"), recursive=True)
        if not files:
            continue

        for f in files:
            try:
                temp_df = pd.read_csv(f, encoding='utf-8-sig')
                if not all(col in temp_df.columns for col in REQUIRED_COLS):
                    continue
                temp_df['class'] = symbol
                df_list.append(temp_df[REQUIRED_COLS + ['class']])
            except Exception as e:
                print(f"파일 로드 실패 {f}: {e}")
    
    if not df_list:
        raise ValueError("유효한 데이터를 로드할 수 없습니다.")
    
    full_df = pd.concat(df_list, ignore_index=True)
    print(f"총 {len(full_df['class'].unique())}개 클래스, {len(full_df)}개 행 로드 완료.")
    return full_df

def rotation_impact_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """1차 분석: RCI, RII 등 기본 영향도 지표를 계산합니다."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[REQUIRED_COLS] = scaler.fit_transform(df[REQUIRED_COLS])

    grouped = df_scaled.groupby('class')
    var_flex = grouped[FLEX_COLS].var().mean(axis=1)
    var_full = grouped[REQUIRED_COLS].var().mean(axis=1)
    delta_var = var_full - var_flex

    centroid_flex = grouped[FLEX_COLS].mean()
    dist_flex = squareform(pdist(centroid_flex, metric='euclidean'))
    
    axis_impacts = {}
    for axis in ROT_COLS:
        centroid_axis = grouped[FLEX_COLS + [axis]].mean()
        dist_axis = squareform(pdist(centroid_axis, metric='euclidean'))
        delta_axis = np.abs(dist_axis - dist_flex).mean(axis=1)
        max_delta = np.max(delta_axis)
        axis_impacts[axis] = delta_axis / max_delta if max_delta > 0 else delta_axis

    df_axis = pd.DataFrame(axis_impacts, index=centroid_flex.index)
    df_axis['축_불균형'] = df_axis.std(axis=1)

    centroid_full = grouped[REQUIRED_COLS].mean()
    dist_full = squareform(pdist(centroid_full, metric='euclidean'))
    delta_dist_class = pd.Series(np.abs(dist_full - dist_flex).mean(axis=1), index=centroid_flex.index)

    delta_var_norm = abs(delta_var / abs(delta_var).max())
    delta_dist_norm = abs(delta_dist_class / abs(delta_dist_class).max())
    rci = 0.5 * (delta_var_norm + delta_dist_norm)
    rii = 0.7 * rci + 0.3 * df_axis['축_불균형']

    result = pd.DataFrame({
        'class': delta_var.index,
        'Δ분산': delta_var,
        'Δ평균거리': delta_dist_class,
        'RCI': rci,
        'Yaw_impact': df_axis['yaw'],
        'Pitch_impact': df_axis['pitch'],
        'Roll_impact': df_axis['roll'],
        '축_불균형': df_axis['축_불균형'],
        'RII': rii
    }).sort_values('RII', ascending=False).reset_index(drop=True)
    
    result.insert(0, '순위', result.index + 1)
    return result

# --- 2단계: RII+ 계산 ---

def compute_RII_plus(df: pd.DataFrame) -> pd.DataFrame:
    """2차 분석: 1차 분석 결과를 바탕으로 RII+ 점수와 최종 해석을 생성합니다."""
    scaler = MinMaxScaler()
    norm_cols = ['Δ분산', 'Δ평균거리', 'Yaw_impact', 'Pitch_impact', 'Roll_impact', '축_불균형', 'RCI']
    df_norm = df.copy()
    df_norm[norm_cols] = scaler.fit_transform(df[norm_cols])

    df_norm['회전_평균'] = df_norm[['Yaw_impact', 'Pitch_impact', 'Roll_impact']].mean(axis=1)

    df_norm['RII_plus'] = (
        0.35 * df_norm['Δ평균거리'] +
        0.25 * df_norm['Δ분산'] +
        0.20 * df_norm['회전_평균'] +
        0.15 * df_norm['축_불균형'] +
        0.05 * df_norm['RCI']
    )

    df_norm = df_norm.sort_values('RII_plus', ascending=False).reset_index(drop=True)
    df_norm['순위'] = df_norm.index + 1

    interpretations = []
    for _, row in df_norm.iterrows():
        rii_plus = row['RII_plus']
        yaw, pitch, roll = row['Yaw_impact'], row['Pitch_impact'], row['Roll_impact']

        if rii_plus >= 0.7: level = "회전이 **결정적 요인**으로 작용"
        elif rii_plus >= 0.4: level = "회전이 **보조적 역할**을 함"
        else: level = "회전이 **거의 영향을 주지 않음**"

        axis = max([('Yaw', yaw), ('Pitch', pitch), ('Roll', roll)], key=lambda x: x[1])[0]
        axis_text = {'Yaw': "좌우 방향(yaw)", 'Pitch': "상하 기울기(pitch)", 'Roll': "손목 비틀림(roll)"}[axis]
        interpretations.append(f"{level}. 주로 **{axis_text}** 변화에 민감한 클래스.")

    df_norm['해석'] = interpretations

    # RII_plus 계산에 사용된 정규화된 값 대신 원본 값으로 결과 테이블 재구성
    final_df = df.merge(df_norm[['class', '순위', 'RII_plus', '해석']], on='class')
    final_df = final_df.sort_values('RII_plus', ascending=False).reset_index(drop=True)
    final_df['순위'] = final_df.index + 1 # 최종 순위 다시 매기기
    
    return final_df[['순위', 'class', 'RII', 'RII_plus', '해석', 'Δ분산', 'Δ평균거리', 'RCI', 'Yaw_impact', 'Pitch_impact', 'Roll_impact', '축_불균형']]

# --- 3단계: 히트맵 시각화 ---
def create_heatmap(df: pd.DataFrame, output_path: str):
    """분석 결과를 바탕으로 회전 영향도 히트맵을 생성하고 저장합니다."""
    print("\n--- 3단계: 히트맵 생성 중 ---")
    
    # 히트맵에 사용할 데이터 선택 및 정렬
    heatmap_data = df.sort_values('RII_plus', ascending=False)
    heatmap_data = heatmap_data.set_index('class')[['Yaw_impact', 'Pitch_impact', 'Roll_impact']]
    
    plt.figure(figsize=(10, 12))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".2f",
        cmap='viridis',
        linewidths=.5,
        cbar_kws={'label': '회전 축별 영향도 (정규화 값)'}
    )
    plt.title('클래스별 회전 축 영향도 분석', fontsize=16, pad=20)
    plt.xlabel('회전 축 (Rotation Axis)', fontsize=12)
    plt.ylabel('클래스 (Class)', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300)
    print(f"히트맵 저장 완료: {output_path}")

# --- 메인 실행 ---

if __name__ == "__main__":
    try:
        # 사용자가 지정한 경로에 output 디렉터리 생성
        output_dir = r'C:\dev\SignGlove-DataAnalysis\Summary statistics\output'
        os.makedirs(output_dir, exist_ok=True)
        print(f"결과물 저장 경로: {output_dir}")

        # --- 1단계 실행 ---
        raw_data = load_data(UNIFIED_DIR)
        print("\n--- 1단계: 기본 회전 영향도(RII) 계산 중 ---")
        rii_result = rotation_impact_scoring(raw_data)
        
        # 1단계 결과 저장
        rii_save_path = os.path.join(output_dir, "rotation_impact_results.csv")
        rii_result.to_csv(rii_save_path, index=False, encoding='utf-8-sig')
        print(f"1단계 결과 저장 완료: {rii_save_path}")

        # --- 2단계 실행 ---
        print("\n--- 2단계: RII+ 점수 및 최종 해석 계산 중 ---")
        rii_plus_result = compute_RII_plus(rii_result)

        # 2단계 최종 결과 저장
        rii_plus_save_path = os.path.join(output_dir, "RII_plus_results.csv")
        rii_plus_result.to_csv(rii_plus_save_path, index=False, encoding='utf-8-sig')
        print(f"2단계 최종 결과 저장 완료: {rii_plus_save_path}")

        # 최종 결과 출력
        print("\n--- 최종 회전 센서 영향 분석 결과 (RII+ ) ---")
        # 보기 좋게 일부 컬럼만 선택하여 출력
        print(rii_plus_result[['순위', 'class', 'RII_plus', '해석']])

        # --- 3단계 실행 ---
        heatmap_save_path = os.path.join(output_dir, "rotation_impact_heatmap.png")
        create_heatmap(rii_plus_result, heatmap_save_path)

    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"\n오류 발생: {e}")