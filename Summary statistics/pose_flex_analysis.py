import os
import glob
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import warnings

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 상수 및 설정 정의 ---

# 1. 경로 설정
UNIFIED_DIR = r"C:\dev\SignGlove-DataAnalysis\unified_v2"
OUTPUT_DIR = r"C:\dev\SignGlove-DataAnalysis\Summary statistics\data_analysis"

# 2. 컬럼 정의
FLEX_COLS = [f'flex{i}' for i in range(1, 6)]
ORIENT_COLS = ['yaw', 'pitch', 'roll']
FULL_COLS = FLEX_COLS + ORIENT_COLS
REQUIRED_COLS = FULL_COLS

# 3. 분석 옵션 토글
USE_STANDARDIZE = True  # z-score 표준화 적용 여부 (기본 True)
SAVE_RAW = True         # 비표준화 지표 추가 저장 여부

# 4. 해석 규칙 정의
HARD = {'var': 0.15, 'dist': 0.10}
SOFT = {'var': 0.05, 'dist': 0.03}

def create_output_dirs():
    """분석 결과를 저장할 폴더를 생성합니다."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"결과 저장 폴더 생성/확인: {OUTPUT_DIR}")

def load_data(path: str) -> pd.DataFrame:
    """
    지정된 경로 하위의 모든 CSV 파일을 찾아 하나의 데이터프레임으로 로드합니다.
    'class' 컬럼을 추가하여 각 데이터의 출처(자음/모음)를 기록합니다.
    """
    print(f"데이터 로드 시작: {path}")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"입력 디렉터리를 찾을 수 없습니다: {path}")

    df_list = []
    symbols_in_path = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    for symbol in symbols_in_path:
        symbol_path = os.path.join(path, symbol)
        files = glob.glob(os.path.join(symbol_path, "**", "*.csv"), recursive=True)
        if not files:
            print(f"경고: '{symbol}' 폴더에 CSV 파일이 없습니다.")
            continue

        for f in files:
            try:
                # 인코딩 문제 대비
                try:
                    df = pd.read_csv(f, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    df = pd.read_csv(f, encoding='cp949')

                # 필수 컬럼 확인
                if not all(col in df.columns for col in REQUIRED_COLS):
                    print(f"경고: 파일에 필수 컬럼이 부족합니다. 건너뜁니다: {f}")
                    continue

                df['class'] = symbol
                df_list.append(df[REQUIRED_COLS + ['class']])
            except Exception as e:
                print(f"파일 로드 실패 {f}: {e}")
                continue

    if not df_list:
        raise ValueError("유효한 데이터를 로드할 수 없습니다.")

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"총 {len(full_df['class'].unique())}개 클래스, {len(full_df)}개 행 로드 완료.")
    return full_df

def compute_metrics(df: pd.DataFrame, use_standardize: bool):
    """데이터프레임으로부터 분산 및 거리 행렬을 계산합니다."""
    
    data = df.copy()
    if use_standardize:
        scaler = StandardScaler()
        data[FULL_COLS] = scaler.fit_transform(data[FULL_COLS])

    grouped = data.groupby('class')
    
    # 클래스별 샘플 수 기록
    n_samples = grouped.size().rename('n_samples')
    
    # 1. 분산 계산
    var_flex = grouped[FLEX_COLS].var().mean(axis=1)
    var_full = grouped[FULL_COLS].var().mean(axis=1)
    
    # 2. 거리 계산
    centroids_flex = grouped[FLEX_COLS].mean()
    centroids_full = grouped[FULL_COLS].mean()
    
    class_order = centroids_flex.index
    
    dist_matrix_flex = squareform(pdist(centroids_flex, metric='euclidean'))
    dist_matrix_full = squareform(pdist(centroids_full, metric='euclidean'))
    
    dist_df_flex = pd.DataFrame(dist_matrix_flex, index=class_order, columns=class_order)
    dist_df_full = pd.DataFrame(dist_matrix_full, index=class_order, columns=class_order)
    
    return {
        "variances": (var_flex, var_full),
        "distances": (dist_df_flex, dist_df_full),
        "n_samples": n_samples,
        "class_order": class_order
    }

def calculate_deltas(metrics):
    """계산된 지표로부터 델타 값을 계산합니다."""
    var_flex, var_full = metrics["variances"]
    dist_df_flex, dist_df_full = metrics["distances"]
    
    delta_variance = var_full - var_flex
    
    dist_diff = (dist_df_full - dist_df_flex).abs()
    np.fill_diagonal(dist_diff.values, 0)
    
    n_classes = len(dist_diff)
    if n_classes > 1:
        delta_distance = dist_diff.sum(axis=1) / (n_classes - 1)
    else:
        delta_distance = pd.Series(0.0, index=dist_diff.index)

    delta_df = pd.DataFrame({
        'delta_variance': delta_variance,
        'delta_distance': delta_distance
    })
    return delta_df

def make_interpretation(delta_df: pd.DataFrame, hard_thresh: dict, soft_thresh: dict) -> pd.Series:
    """델타 값에 기반하여 자동 해석을 생성합니다."""
    interpretations = []
    for _, row in delta_df.iterrows():
        dv = row['delta_variance']
        dd = row['delta_distance']
        
        if dv > hard_thresh['var'] or dd > hard_thresh['dist']:
            interp = "회전값이 결정적으로 중요"
        elif dv > soft_thresh['var'] or dd > soft_thresh['dist']:
            interp = "약간의 회전 영향 있음"
        elif dv < 0 and dd <= 0:
            interp = "flex만으로 매우 안정적"
        else:
            interp = "flex만으로 충분"
        interpretations.append(interp)
        
    return pd.Series(interpretations, index=delta_df.index, name='interpretation')

def main():
    """메인 분석 로직"""
    create_output_dirs()
    
    try:
        full_data = load_data(UNIFIED_DIR)
    except (FileNotFoundError, ValueError) as e:
        print(f"데이터 처리 중단: {e}")
        return

    print("\n--- 표준화(Standardized) 데이터로 지표 계산 ---")
    metrics_std = compute_metrics(full_data, use_standardize=True)
    deltas_std = calculate_deltas(metrics_std)
    
    results_df = deltas_std.copy()
    
    if SAVE_RAW:
        print("\n--- 비표준화(Raw) 데이터로 지표 계산 ---")
        metrics_raw = compute_metrics(full_data, use_standardize=False)
        deltas_raw = calculate_deltas(metrics_raw)
        
        deltas_raw.rename(columns={
            'delta_variance': 'delta_variance_raw',
            'delta_distance': 'delta_distance_raw'
        }, inplace=True)
        results_df = pd.concat([results_df, deltas_raw], axis=1)

    interpretation = make_interpretation(deltas_std, HARD, SOFT)
    results_df['interpretation'] = interpretation
    
    results_df['n_samples'] = metrics_std['n_samples']
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'class'}, inplace=True)

    overview_list = []
    
    print("\n--- 클래스별 결과 저장 ---")
    for symbol in metrics_std["class_order"]:
        symbol_result = results_df[results_df['class'] == symbol]
        
        save_path = os.path.join(OUTPUT_DIR, f"{symbol}_delta_stats.csv")
        symbol_result.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"저장 완료: {save_path}")
        
        overview_list.append(symbol_result)
        
    if overview_list:
        overview_df = pd.concat(overview_list, ignore_index=True)
        overview_path = os.path.join(OUTPUT_DIR, "delta_stats_overview.csv")
        overview_df.to_csv(overview_path, index=False, encoding='utf-8-sig')
        print(f"\n전체 요약 저장 완료: {overview_path}")
    else:
        print("\n처리된 데이터가 없어 전체 요약 파일을 생성하지 않습니다.")

    print("\n분석 완료.")

if __name__ == "__main__":
    main()