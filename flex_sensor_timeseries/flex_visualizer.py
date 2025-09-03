import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

def make_plots_from_folder(input_dir, output_dir="/mnt/data"):
    """
    지정한 폴더 안의 CSV 파일들을 자동으로 그룹핑해서
    ㄱ_1.png, ㄱ_2.png ... 형태의 그래프 생성
    """

    # 1) 폴더 내 모든 CSV 파일 가져오기
    files = glob(os.path.join(input_dir, "*.csv"))

    # 2) 파일명에서 그룹과 번호 추출하여 dict에 저장
    grouped = {}
    for f in files:
        base = os.path.basename(f)
        try:
            # 파일명에서 마지막 두 부분 (예: ㄱ, 1.csv) 추출
            group, num_ext = base.split("_")[-2:]
            num = num_ext.replace(".csv", "")
            key = (group, num)   # ex) ('ㄱ', '1')
            grouped.setdefault(key, []).append(f)
        except Exception as e:
            print(f"⚠️ 파일명 인식 실패: {base} ({e})")

    # 3) 그룹별 y축 범위 계산
    y_ranges = {}
    for (group, num), file_list in grouped.items():
        all_values = []
        for f in file_list:
            df = pd.read_csv(f)
            for col in ["flex1", "flex2", "flex3", "flex4", "flex5"]:
                if col in df.columns:
                    all_values.extend(df[col].values)
        if group not in y_ranges:
            y_ranges[group] = [min(all_values), max(all_values)]
        else:
            y_ranges[group][0] = min(y_ranges[group][0], min(all_values))
            y_ranges[group][1] = max(y_ranges[group][1], max(all_values))

    # 4) 그래프 생성
    for (group, num), file_list in grouped.items():
        dfs = [pd.read_csv(f) for f in file_list]
        data = pd.concat(dfs, ignore_index=True)

        plt.figure(figsize=(15, 6))
        for col in ["flex1", "flex2", "flex3", "flex4", "flex5"]:
            if col in data.columns:
                plt.plot(data[col], label=col)

        plt.title(f"Flex Sensor Data - {group}_{num}")
        plt.xlabel("Sample index")
        plt.ylabel("ADC Value")
        plt.ylim(y_ranges[group][0], y_ranges[group][1])  # 그룹 y축 통일
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        out_path = os.path.join(output_dir, f"{group}_{num}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved: {out_path}")
