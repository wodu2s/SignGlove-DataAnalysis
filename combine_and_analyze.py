import pandas as pd
import os
import traceback

# --- 설정: 여기에 입력 폴더와 출력 파일 경로를 직접 지정하세요 ---
INPUT_DIRECTORY = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\data cleaning\backup\ㄱ_20250901_074541\1"
OUTPUT_FILEPATH = r"C:\Users\jjy06\clac_stats.csv"
# ---------------------------------------------------------

def combine_and_analyze():
    """
    지정된 폴더의 모든 CSV 파일을 하나로 합치고, 숫자 열에 대한 기술 통계를 계산하여
    결과를 새로운 CSV 파일로 저장합니다.
    """
    try:
        print(f"--- 스크립트 실행 시작 ---")
        print(f"입력 폴더: {INPUT_DIRECTORY}")

        if not os.path.isdir(INPUT_DIRECTORY):
            print(f"오류: 입력 경로 '{INPUT_DIRECTORY}'가 올바른 폴더가 아닙니다.")
            return

        csv_files = [f for f in os.listdir(INPUT_DIRECTORY) if f.lower().endswith('.csv')]
        if not csv_files:
            print(f"오류: '{INPUT_DIRECTORY}' 폴더에서 CSV 파일을 찾을 수 없습니다.")
            return

        print(f"총 {len(csv_files)}개의 CSV 파일을 찾았습니다. 파일을 결합하고 분석합니다...")

        df_list = []
        for filename in csv_files:
            full_path = os.path.join(INPUT_DIRECTORY, filename)
            try:
                df_list.append(pd.read_csv(full_path))
            except Exception as e:
                print(f"'{filename}' 파일을 읽는 중 오류 발생: {e}")
        
        if not df_list:
            print("오류: CSV 파일들을 성공적으로 읽어오지 못했습니다.")
            return

        combined_df = pd.concat(df_list, ignore_index=True)
        print("--- 모든 CSV 파일이 하나의 데이터프레임으로 결합되었습니다. ---")

        numerical_df = combined_df.select_dtypes(include=['number'])
        if numerical_df.empty:
            print("오류: 결합된 데이터에 숫자 데이터가 없습니다.")
            return

        print("--- 통계 계산 중... ---")
        stats_df = numerical_df.describe()
        stats_df.loc['variance'] = numerical_df.var()

        stats_df.to_csv(OUTPUT_FILEPATH, encoding='utf-8-sig')
        
        print(f"--- 성공! 모든 통계가 '{OUTPUT_FILEPATH}' 파일에 저장되었습니다. ---")

    except Exception as e:
        print(f"스크립트 실행 중 예기치 않은 오류가 발생했습니다: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    combine_and_analyze()