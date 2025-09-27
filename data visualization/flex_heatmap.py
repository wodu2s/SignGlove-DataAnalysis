# -*- coding: utf-8 -*-
"""
센서 데이터 상관관계 히트맵 생성 스크립트

[기능]
- 단일 CSV 파일 또는 여러 에피소드 폴더의 CSV들을 병합하여 데이터 로드
- 특정 접두사(prefix)나 추가 열(extra-cols)을 기준으로 분석 대상 열 선택
- Z-score 표준화, 분산이 0인 열 제거 등 전처리 옵션 제공
- Pearson 또는 Spearman 상관계수 계산 방식 선택
- Seaborn을 사용한 히트맵 및 클러스터맵 생성 및 저장
- 상세한 로깅 및 커맨드 라인 인터페이스(CLI) 지원

[모듈로 사용 시]
- 필요한 함수를 import하여 개별적으로 사용 가능합니다.

[CLI로 사용 시]
- 단일 파일 모드:
  python visualize_corr.py --csv "C:\data\sample.csv" --symbol ㄱ --output_root "C:\out"

- 폴더 모드:
  python visualize_corr.py --root "C:\data\ㄷ_20250901_080751" --episodes 1 2 3 4 5 --symbol ㄷ --output_root "C:\out" --extra-cols AX1 AY1 --method spearman

- 인자 없이 기본값 사용 모드:
  python visualize_corr.py
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] - %(message)s",
    stream=sys.stdout
)

# --- 설정 상수 ---
# argparse 인자가 없을 경우 사용될 기본값들
DEFAULT_SYMBOL_DIR = Path(r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data cleaning\backup\ㄷ_20250901_080751")
DEFAULT_SYMBOL = "ㄷ"
EPISODE_DIRS = ["1", "2", "3", "4", "5"]
NA_VALUES = ["", " ", "NA"]
OUTPUT_ROOT = Path(r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data visualization\visualization")

# --- 함수 정의 ---

def load_single_csv(path: Path) -> pd.DataFrame | None:
    """단일 CSV 파일을 로드합니다."""
    if not path.is_file():
        logging.error(f"CSV 파일을 찾을 수 없습니다: {path}")
        return None
    logging.info(f"단일 CSV 파일 로드 중: {path}")
    try:
        return pd.read_csv(path, na_values=NA_VALUES)
    except Exception as e:
        logging.error(f"CSV 파일 로드 중 오류 발생: {e}")
        return None

def load_from_root(root: Path, episodes: list[str]) -> pd.DataFrame | None:
    """루트 폴더 내 각 에피소드 폴더의 모든 CSV를 병합하여 로드합니다."""
    all_dfs = []
    logging.info(f"폴더 모드로 데이터 로드 시작. 루트: {root}")
    if not root.is_dir():
        logging.error(f"지정된 데이터 루트 디렉터리를 찾을 수 없습니다: {root}")
        return None

    for episode in episodes:
        episode_path = root / episode
        if not episode_path.is_dir():
            logging.warning(f"에피소드 폴더를 찾을 수 없습니다: {episode_path}")
            continue

        csv_files = list(episode_path.glob("*.csv"))
        if not csv_files:
            logging.warning(f"에피소드 '{episode}' 폴더에 CSV 파일이 없습니다.")
            continue

        logging.info(f"에피소드 '{episode}'에서 {len(csv_files)}개의 CSV 파일 로드 중...")
        for file in csv_files:
            try:
                df = pd.read_csv(file, na_values=NA_VALUES)
                df["episode"] = episode  # 원본 구분을 위해 episode 열 주입
                all_dfs.append(df)
            except Exception as e:
                logging.error(f"파일 로드 실패 '{file}': {e}")

    if not all_dfs:
        logging.error("병합할 데이터가 없습니다. 폴더 구조와 파일 내용을 확인하세요.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)
    logging.info(f"총 {len(combined_df)}개의 행으로 데이터 병합 완료.")
    return combined_df

def select_sensor_columns(df: pd.DataFrame, prefix: str, extra_cols: list[str] | None) -> list[str]:
    """분석할 센서 데이터 열을 선택합니다."""
    selected = set()
    # 1. 접두사로 시작하는 열 선택
    if prefix:
        prefix_cols = {col for col in df.columns if str(col).startswith(prefix)}
        selected.update(prefix_cols)
        logging.info(f"'{prefix}' 접두사로 {len(prefix_cols)}개 열 선택.")

    # 2. 추가로 지정된 열 선택
    if extra_cols:
        # 실제 데이터프레임에 있는 열만 추가
        valid_extra = [col for col in extra_cols if col in df.columns]
        selected.update(valid_extra)
        logging.info(f"추가 지정된 열 {len(valid_extra)}개 선택: {valid_extra}")

    final_cols = sorted(list(selected))
    logging.info(f"최종 분석 대상 열 ({len(final_cols)}개): {final_cols}")
    return final_cols

def preprocess_numeric(
    df: pd.DataFrame, cols: list[str], zscore: bool, drop_nonvar: bool
) -> tuple[pd.DataFrame, list[str]]:
    """데이터를 숫자형으로 변환하고, 옵션에 따라 전처리를 수행합니다."""
    logging.info("전처리 시작: 숫자형 변환 및 옵션 적용")
    df_processed = df.copy()

    # 1. 숫자형으로 변환 (변환 불가 시 NaN 처리)
    df_processed[cols] = df_processed[cols].apply(pd.to_numeric, errors='coerce')
    initial_na = df_processed[cols].isna().sum().sum()
    logging.info(f"숫자형 변환 후 총 {initial_na}개의 결측치(NaN) 발생.")

    # 2. 분산이 0인 열(상수열) 제거
    if drop_nonvar:
        variances = df_processed[cols].var()
        non_var_cols = variances[variances == 0].index.tolist()
        if non_var_cols:
            df_processed = df_processed.drop(columns=non_var_cols)
            # cols 리스트도 동기화
            cols = [c for c in cols if c not in non_var_cols]
            logging.info(f"분산이 0인 상수열 제거: {non_var_cols}")

    # 3. Z-score 표준화
    if zscore:
        logging.info("Z-score 표준화 적용 중...")
        for col in cols:
            mean = df_processed[col].mean()
            std = df_processed[col].std()
            if std > 0:
                df_processed[col] = (df_processed[col] - mean) / std
            else:
                df_processed[col] = 0  # 표준편차가 0이면 모든 값을 0으로

    return df_processed, cols

def compute_corr(
    df: pd.DataFrame, cols: list[str], method: str, by: str
) -> pd.DataFrame | None:
    """상관계수 행렬을 계산합니다."""
    logging.info(f"'{by}' 기준으로 상관계수 계산 시작 (방식: {method}).")
    
    # episode_mean 모드일 경우, 먼저 episode별 평균 집계
    if by == "episode_mean":
        if "episode" not in df.columns:
            logging.error("--by episode_mean 옵션은 폴더 모드에서만 유효합니다.")
            return None
        data_to_corr = df.groupby("episode")[cols].mean()
        logging.info(f"Episode별 평균 집계 완료. {len(data_to_corr)}개 행으로 축소.")
    else: # 'overall' 모드
        data_to_corr = df[cols]

    # 상관계수 계산 (pandas가 내부적으로 pairwise로 NA 처리)
    corr_matrix = data_to_corr.corr(method=method)
    return corr_matrix

def plot_heatmap(
    corr: pd.DataFrame, out_png: Path, mask_upper: bool, figsize: tuple, dpi: int
):
    """상관계수 히트맵을 생성하고 저장합니다."""
    logging.info(f"히트맵 생성 시작. 저장 경로: {out_png}")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=figsize)

    mask = None
    if mask_upper:
        # 상삼각 행렬을 가리기 위한 마스크 생성
        mask = np.triu(np.ones_like(corr, dtype=bool))
        logging.info("히트맵 상삼각 행렬 마스킹 적용.")

    sns.heatmap(
        corr,
        annot=True,          # 각 셀에 값 표시
        fmt=".2f",           # 소수점 둘째 자리까지
        cmap="RdBu_r",       # Red-Blue diverging 컬러맵 (중앙 0)
        vmin=-1, vmax=1,     # 컬러바 범위 고정
        center=0,            # 0을 중앙 색상으로 설정
        mask=mask,
        ax=ax
    )

    ax.set_title("Sensor Correlation Heatmap", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # 파일 저장
    try:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
        logging.info(f"히트맵 저장 완료: {out_png}")
    except Exception as e:
        logging.error(f"히트맵 저장 실패: {e}")
    finally:
        plt.close(fig)

def maybe_clustermap(
    corr: pd.DataFrame, out_png: Path, figsize: tuple, dpi: int, enabled: bool
):
    """옵션이 활성화된 경우 클러스터맵을 생성하고 저장합니다."""
    if not enabled:
        return

    # 클러스터맵용 파일명 생성
    clustermap_path = out_png.with_name(f"{out_png.stem}_clustered.png")
    logging.info(f"클러스터맵 생성 시작. 저장 경로: {clustermap_path}")

    try:
        g = sns.clustermap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            vmin=-1, vmax=1,
            figsize=figsize
        )
        g.savefig(clustermap_path, dpi=dpi, bbox_inches='tight')
        logging.info(f"클러스터맵 저장 완료: {clustermap_path}")
    except Exception as e:
        logging.error(f"클러스터맵 생성 또는 저장 실패: {e}")
    finally:
        plt.close('all')

def main():
    """메인 실행 함수: CLI 인자 파싱 및 전체 파이프라인 실행"""
    parser = argparse.ArgumentParser(description="센서 데이터 상관관계 히트맵 생성기")
    
    # 입력 모드 (상호 배타적)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--csv", type=Path, help="단일 CSV 파일 경로")
    input_group.add_argument("--root", type=Path, help="에피소드 폴더들을 포함하는 루트 디렉터리")

    # 폴더 모드 전용 옵션
    parser.add_argument("--episodes", nargs='+', default=['1', '2', '3', '4', '5'], help="루트 폴더 내 분석할 에피소드 목록")

    # 공통 옵션
    parser.add_argument("--symbol", type=str, help="결과물 파일명/경로에 사용할 심볼 (예: ㄱ, ㄷ)")
    parser.add_argument("--output_root", type=Path, help="결과를 저장할 최상위 디렉터리")
    parser.add_argument("--sensor-prefix", type=str, default="flex", help="분석할 열의 접두사")
    parser.add_argument("--extra-cols", nargs='*', help="접두사 외에 추가로 분석에 포함할 열 목록")

    # 전처리 옵션
    parser.add_argument("--drop-nonvar", action='store_true', help="분산이 0인 상수열을 제거합니다.")
    parser.add_argument("--zscore", action='store_true', help="데이터를 Z-score로 표준화합니다.")
    parser.add_argument("--by", choices=['overall', 'episode_mean'], default='overall', help="상관계수 계산 기준")

    # 계산 및 시각화 옵션
    parser.add_argument("--method", choices=['pearson', 'spearman'], default='pearson', help="상관계수 계산 방식")
    parser.add_argument("--mask-upper", dest='mask_upper', action='store_true', help="히트맵의 상삼각 행렬을 가립니다 (기본값).")
    parser.add_argument("--no-mask-upper", dest='mask_upper', action='store_false', help="히트맵의 상삼각 행렬을 가리지 않습니다.")
    parser.add_argument("--figsize", type=int, nargs=2, default=[8, 6], help="그림 크기 (너비 높이)")
    parser.add_argument("--dpi", type=int, default=150, help="저장 이미지의 DPI")
    parser.add_argument("--cluster", action='store_true', help="기본 히트맵 외에 클러스터맵도 함께 생성합니다.")
    parser.set_defaults(mask_upper=True)

    args = parser.parse_args()

    # 1. 데이터 로드
    df = None
    if args.csv:
        # 단일 파일 모드
        df = load_single_csv(args.csv)
    elif args.root:
        # 사용자 지정 루트 모드
        df = load_from_root(args.root, args.episodes)
    else:
        # 기본값 사용 모드
        logging.info("입력 인자가 없어 DEFAULT_SYMBOL_DIR을 사용합니다.")
        logging.info(f"기본 데이터 경로 사용: {DEFAULT_SYMBOL_DIR}")
        logging.info(f"기본 심볼: {DEFAULT_SYMBOL}")
        df = load_from_root(DEFAULT_SYMBOL_DIR, EPISODE_DIRS)
        # 기본값 사용 시 args 객체에 기본값 설정
        if args.symbol is None: args.symbol = DEFAULT_SYMBOL
        if args.output_root is None: args.output_root = OUTPUT_ROOT
        if args.episodes == ['1', '2', '3', '4', '5']: args.episodes = EPISODE_DIRS # 기본값과 동일하면 덮어쓰기

    if df is None:
        logging.critical("데이터 로드 실패. 프로그램을 종료합니다.")
        return

    # 2. 분석 대상 열 선택
    sensor_cols = select_sensor_columns(df, args.sensor_prefix, args.extra_cols)
    if len(sensor_cols) < 2:
        logging.critical(f"분석할 열이 2개 미만입니다({len(sensor_cols)}개). 상관계수를 계산할 수 없습니다. --sensor-prefix 나 --extra-cols 인자를 확인하세요.")
        return

    # 3. 전처리
    df, sensor_cols = preprocess_numeric(df, sensor_cols, args.zscore, args.drop_nonvar)
    if len(sensor_cols) < 2:
        logging.critical(f"전처리 후 분석할 열이 2개 미만입니다({len(sensor_cols)}개). 상관계수를 계산할 수 없습니다.")
        return

    # 4. 상관계수 계산
    corr_matrix = compute_corr(df, sensor_cols, args.method, args.by)
    if corr_matrix is None or corr_matrix.empty:
        logging.critical("상관계수 행렬을 계산할 수 없습니다. 데이터에 유효한 숫자 값이 충분한지 확인하세요.")
        return

    # 5. 결과 저장 경로 설정
    output_dir = args.output_root / args.symbol / "Heatmap"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"Correlation_Heatmap_{args.symbol}_{args.by}_{args.method}.png"
    output_png_path = output_dir / file_name

    # 6. 시각화 및 저장
    plot_heatmap(corr_matrix, output_png_path, args.mask_upper, tuple(args.figsize), args.dpi)
    maybe_clustermap(corr_matrix, output_png_path, tuple(args.figsize), args.dpi, args.cluster)

    logging.info("모든 작업이 성공적으로 완료되었습니다.")

if __name__ == "__main__":
    main()