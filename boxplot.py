import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 불러오기 (파일 경로를 본인 환경에 맞게 수정하세요)
file_path = r"C:\Users\jjy06\OneDrive\바탕 화면\DeepBot\AIoT 프로젝트\Data Analysis\cleaned_dataset.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# flex4 데이터 추출
flex4_data = df["flex4"]

# 박스플롯 생성
plt.figure(figsize=(6, 8))
plt.boxplot(flex4_data, vert=True, patch_artist=True)

# 그래프 꾸미기
plt.title("Flex 4 Boxplot")
plt.ylabel("Flex 4 Values")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 그래프 출력
plt.show()
