import pandas as pd
import os
import glob

# 처리할 파일이 있는 폴더 경로
input_dir = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\unified_new\ㄱ"

# 처리된 파일을 저장할 폴더 경로
output_dir = os.path.join(input_dir, "cleaned_files")
os.makedirs(output_dir, exist_ok=True)

# 폴더 내 모든 CSV 파일 목록 가져오기
csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

if not csv_files:
    print(f"'{input_dir}' 경로에 CSV 파일이 없습니다.")
else:
    print(f"총 {len(csv_files)}개의 CSV 파일을 처리합니다.")

    # 각 CSV 파일에 대해 반복 작업
    for file_path in csv_files:
        try:
            print(f"\n--- 처리 중: {os.path.basename(file_path)} ---")
            
            # 데이터 불러오기
            df = pd.read_csv(file_path)

            # 삭제할 행 번호 (15번째 행은 인덱스 14)
            row_to_drop = 14

            # 행 삭제
            if row_to_drop < len(df):
                df = df.drop(row_to_drop, axis=0).reset_index(drop=True)
                print(f"- {row_to_drop + 1}번째 행을 삭제했습니다.")
            else:
                print(f"- 파일에 {row_to_drop + 1}번째 행이 없어 삭제를 건너뜁니다.")

            # 새 파일명 생성
            base_name = os.path.basename(file_path)
            output_filename = f"cleaned_{base_name}"
            output_path = os.path.join(output_dir, output_filename)

            # CSV 파일로 저장
            df.to_csv(output_path, index=False)
            print(f"파일 저장 완료: {output_path}")

        except Exception as e:
            print(f"오류 발생: {os.path.basename(file_path)} 처리 중 오류가 발생했습니다. ({e})")