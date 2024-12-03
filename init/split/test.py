import pandas as pd
import os


def excel_sheets_to_csv(input_file):
    try:
        # 读取Excel的所有sheet
        excel_file = pd.ExcelFile(input_file)

        # 获取所有sheet名称
        sheet_names = excel_file.sheet_names

        # 遍历处理每个sheet
        for i, sheet in enumerate(sheet_names):
            if i >= 5:  # 只处理前5个sheet
                break

            # 读取当前sheet
            df = pd.read_excel(input_file, sheet_name=sheet)

            # 构造输出文件名
            output_file = f"init_state_{i}.csv"

            # 保存为CSV
            df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f'已将Sheet "{sheet}" 保存为: {output_file}')

        print("转换完成！")

    except Exception as e:
        print(f"转换失败: {str(e)}")


if __name__ == "__main__":
    input_file = input("请输入Excel文件路径: ")
    excel_sheets_to_csv(input_file)
