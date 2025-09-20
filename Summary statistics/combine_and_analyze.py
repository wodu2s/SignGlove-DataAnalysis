import os
import sys
import glob
import pandas as pd
import logging

# --- 설정 상수 ---
# 분석할 고정 심볼 경로와 이름 설정
DEFAULT_SYMBOL_DIR = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data cleaning\backup\ㄷ_20250901_080751"
DEFAULT_SYMBOL = "ㄷ"

# 에피소드 폴더 목록
EPISODE_DIRS = ["1", "2", "3", "4", "5"]
# 통계 지표 순서
STATS_INDEX_ORDER = ['sum', 'mean', 'var', 'std', 'median', 'min', 'max', 'q1', 'q3']
# 결측치로 간주할 값 목록
NA_VALUES = ["", " ", "NA"]

# 생성된 파일을 저장할 디렉터리
OUTPUT_DIR = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\Summary statistics\summary data"

def setup_logging(target_char):
    """타깃 기호별로 로그 파일 설정을 초기화합니다."""
    log_filename = f"run_log_{target_char}.txt"
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def find_csv_files(episode_path):
    """주어진 에피소드 경로에서 CSV 파일 목록을 찾습니다."""
    if not os.path.isdir(episode_path):
        logging.error(f"에피소드 디렉터리를 찾을 수 없습니다: {episode_path}")
        return None
    
    csv_files = glob.glob(os.path.join(episode_path, '*.csv'))
    if not csv_files:
        logging.error(f"에피소드에 CSV 파일이 없습니다: {episode_path}")
        return None
        
    return csv_files

def check_for_missing_values(df, filepath, episode, numeric_cols):
    """데이터프레임의 수치형 컬럼에서 결측치를 확인합니다."""
    missing_reports = []
    for col in numeric_cols:
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        missing_count = numeric_series.isnull().sum()
        
        if missing_count > 0:
            report = {
                'filepath': filepath,
                'episode': episode,
                'column': col,
                'missing_count': missing_count
            }
            missing_reports.append(report)
            logging.warning(f"결측치 발견: 파일='{os.path.basename(filepath)}', 컬럼='{col}', 개수={missing_count}")
            
    return missing_reports

def load_and_merge_episode_data(target_char, episode, output_dir):
    """특정 타깃의 한 에피소드에 속한 모든 CSV를 병합하고 검증합니다."""
    # 경로 구성을 DEFAULT_SYMBOL_DIR 기준으로 변경
    episode_path = os.path.join(DEFAULT_SYMBOL_DIR, episode)
    csv_files = find_csv_files(episode_path)
    if csv_files is None:
        return None, None

    all_missing_reports = []
    episode_dfs = []
    
    try:
        first_df = pd.read_csv(csv_files[0], encoding='utf-8', na_values=NA_VALUES)
        numeric_cols = first_df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            logging.error(f"파일에 수치형 데이터 컬럼이 없습니다: {csv_files[0]}")
            return None, None
    except Exception as e:
        logging.error(f"첫 번째 CSV 파일을 읽는 중 오류 발생: {csv_files[0]}, 오류: {e}")
        return None, None

    for f in csv_files:
        try:
            df = pd.read_csv(f, encoding='utf-8', na_values=NA_VALUES)
            df = df.reindex(columns=first_df.columns)
            
            missing_reports = check_for_missing_values(df, f, episode, numeric_cols)
            if missing_reports:
                all_missing_reports.extend(missing_reports)
            episode_dfs.append(df)
        except Exception as e:
            logging.error(f"CSV 파일을 읽는 중 오류 발생: {f}, 오류: {e}")
            return None, None

    if all_missing_reports:
        report_df = pd.DataFrame(all_missing_reports)
        report_filename = f"missing_report_{target_char}.csv"
        report_filepath = os.path.join(output_dir, report_filename)
        report_df.to_csv(report_filepath, index=False, encoding='utf-8-sig')
        logging.error(f"결측치가 발견되어 처리를 중단합니다. '{report_filepath}' 파일을 확인하세요.")
        return None, None

    merged_df = pd.concat(episode_dfs, ignore_index=True)
    return merged_df, numeric_cols

def calculate_statistics(df, numeric_cols):
    """수치형 컬럼에 대한 통계 지표를 계산합니다."""
    desc = df[numeric_cols].describe(percentiles=[.25, .75]).round(4)
    stats = {
        'sum': df[numeric_cols].sum().round(4),
        'var': df[numeric_cols].var().round(4),
        'median': df[numeric_cols].median().round(4),
    }
    summary_df = pd.DataFrame(stats).transpose()
    desc = desc.rename(index={'25%': 'q1', '50%': 'median_desc', '75%': 'q3'})
    final_summary = pd.concat([summary_df, desc.drop('median_desc')])
    if 'count' in final_summary.index:
        final_summary = final_summary.drop('count')
    final_summary = final_summary.reindex(STATS_INDEX_ORDER)
    return final_summary

def process_target(target_char):
    """하나의 타깃 기호에 대한 전체 처리 과정을 수행합니다."""
    symbol_specific_output_dir = os.path.join(OUTPUT_DIR, target_char)
    os.makedirs(symbol_specific_output_dir, exist_ok=True)
    setup_logging(target_char)
    logging.info(f"--- 타깃 '{target_char}' 처리 시작 ---")
    
    # 경로 검사를 DEFAULT_SYMBOL_DIR 기준으로 변경
    if not os.path.isdir(DEFAULT_SYMBOL_DIR):
        logging.error(f"타깃 디렉터리를 찾을 수 없습니다: {DEFAULT_SYMBOL_DIR}")
        return

    all_episodes_df_list = []
    final_numeric_cols = None

    for episode in EPISODE_DIRS:
        logging.info(f">> 에피소드 '{episode}' 처리 중...")
        episode_df, numeric_cols = load_and_merge_episode_data(target_char, episode, symbol_specific_output_dir)
        if episode_df is None:
            return 

        if final_numeric_cols is None:
            final_numeric_cols = numeric_cols
        
        episode_summary = calculate_statistics(episode_df, final_numeric_cols)
        episode_filename = f"summary_data_{target_char}_{episode}.csv"
        episode_filepath = os.path.join(symbol_specific_output_dir, episode_filename)
        episode_summary.to_csv(episode_filepath, encoding='utf-8-sig')
        logging.info(f"에피소드 요약 파일 저장 완료: {episode_filepath}")
        all_episodes_df_list.append(episode_df)

    if not all_episodes_df_list:
        logging.warning("처리할 에피소드 데이터가 없어 전체 통합 요약을 생성할 수 없습니다.")
        return
        
    logging.info(">> 모든 에피소드 통합 처리 중...")
    total_df = pd.concat(all_episodes_df_list, ignore_index=True)
    total_summary = calculate_statistics(total_df, final_numeric_cols)
    total_filename = f"summary_data_{target_char}.csv"
    total_filepath = os.path.join(symbol_specific_output_dir, total_filename)
    total_summary.to_csv(total_filepath, encoding='utf-8-sig')
    logging.info(f"전체 통합 요약 파일 저장 완료: {total_filepath}")
    logging.info(f"--- 타깃 '{target_char}' 처리 완료 ---")

def main():
    """메인 실행 함수"""
    # 설정된 심볼 디렉터리가 존재하는지 확인
    if not os.path.isdir(DEFAULT_SYMBOL_DIR):
        print(f"오류: 설정된 심볼 디렉터리를 찾을 수 없습니다: {DEFAULT_SYMBOL_DIR}", file=sys.stderr)
        sys.exit(1)
    
    print(f"지정된 타깃으로 처리 시작: {DEFAULT_SYMBOL}")
    # 고정된 심볼로 process_target 함수 호출
    process_target(DEFAULT_SYMBOL)

if __name__ == '__main__':
    main()