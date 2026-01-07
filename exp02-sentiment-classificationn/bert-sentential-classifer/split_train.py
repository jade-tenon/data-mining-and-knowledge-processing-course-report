import pandas as pd
import os

# 读取原始数据
df = pd.read_csv('dataset/train.csv', header=None)

# 分割为5份
parts = []
size = len(df) // 5
for i in range(5):
    start = i * size
    end = (i + 1) * size if i < 4 else len(df)
    part = df.iloc[start:end]
    part.to_csv(f'dataset/train_part{i+1}.csv', index=False, header=False)
    parts.append(f'train_part{i+1}.csv')

print("分割完成，文件名：", parts)

# 复制一份约 5MB 的小型训练集，保存为 dataset/small_train.csv
src_path = 'dataset/train_part5.csv'
df = pd.read_csv(src_path, header=None)

# 目标大小（字节）
target_size = 5 * 1024 * 1024  # 5MB
rows = []
current_size = 0
for idx, row in df.iterrows():
    # 将当前行转为csv字符串
    line = ','.join([str(x) for x in row]) + '\n'
    line_bytes = line.encode('utf-8')
    if current_size + len(line_bytes) > target_size:
        break
    rows.append(row)
    current_size += len(line_bytes)

small_df = pd.DataFrame(rows)
small_df.to_csv('dataset/small_train.csv', index=False, header=False, encoding='utf-8')
print(f"已生成小型训练集 dataset/small_train.csv，大小约 {current_size/1024/1024:.2f} MB，共 {len(rows)} 行。")
