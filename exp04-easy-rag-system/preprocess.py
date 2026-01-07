# python
# File: `exp04-easy-rag-system/preprocess.py`
import os
import json
from bs4 import BeautifulSoup
import re

def extract_text_and_title_from_html(html_filepath):
    try:
        with open(html_filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()

        soup = BeautifulSoup(html_content, 'lxml')

        title_tag = soup.find('title')
        title_string = title_tag.string if title_tag else None
        title = title_string.strip() if title_string else os.path.basename(html_filepath)
        title = title.replace('.html', '')

        content_tag = soup.find('content')
        if not content_tag:
            content_tag = soup.find('div', class_='rich_media_content')
        if not content_tag:
            content_tag = soup.find('article')
        if not content_tag:
            content_tag = soup.find('main')
        if not content_tag:
            content_tag = soup.find('body')

        if content_tag:
            text = content_tag.get_text(separator='\n', strip=True)
            text = re.sub(r'\n\s*\n', '\n', text).strip()
            text = text.replace('阅读原文', '').strip()
            return title, text
        else:
            print(f"警告：在文件 {html_filepath} 中未找到明确的正文标签。")
            return title, None

    except FileNotFoundError:
        print(f"错误：文件 {html_filepath} 未找到。")
        return None, None
    except Exception as e:
        print(f"处理文件 {html_filepath} 时出错: {e}")
        return None, None

def split_text(text, chunk_size=350, chunk_overlap=50):
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text):
            break
        if start < chunk_size and len(chunks) > 1 and chunks[-1] == chunks[-2][chunk_size - chunk_overlap:]:
            chunks.pop()
            start = len(text)

    if start < len(text) and start > 0:
        last_chunk = text[start - chunk_size + chunk_overlap:]
        if chunks and last_chunk != chunks[-1]:
            if not chunks[-1].endswith(last_chunk):
                chunks.append(last_chunk)
        elif not chunks:
            chunks.append(last_chunk)

    return [c.strip() for c in chunks if c.strip()]

def truncate_by_bytes(s, max_bytes, encoding='utf-8'):
    """按字节长度截断字符串，保证返回字符串编码后不超过 max_bytes。"""
    if s is None:
        return s
    if max_bytes is None:
        return s
    encoded = s.encode(encoding)
    if len(encoded) <= max_bytes:
        return s
    # 二分查找最大字符数前缀，使得编码后长度 <= max_bytes
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(s[:mid].encode(encoding)) <= max_bytes:
            lo = mid
        else:
            hi = mid - 1
    return s[:lo]

# --- 配置（硬编码最大字节数为 400） ---
html_directory = './data/'
output_json_path = './data/processed_data.json'
CHUNK_SIZE = 350  # 字符分块大小（保守）
CHUNK_OVERLAP = 50

BYTE_LIMIT = 350  # 最大允许字节数（强制为 400 bytes）

all_data_for_milvus = []
file_count = 0
chunk_count = 0
truncated_examples = 0

print(f"开始处理目录 '{html_directory}' 中的 HTML 文件...")

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

html_files = [f for f in os.listdir(html_directory) if f.endswith('.html')]
print(f"找到 {len(html_files)} 个 HTML 文件。")

max_bytes = int(BYTE_LIMIT)

for filename in html_files:
    filepath = os.path.join(html_directory, filename)
    print(f"  处理文件: {filename} ...")
    file_count += 1

    title, main_text = extract_text_and_title_from_html(filepath)

    if main_text:
        chunks = split_text(main_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        print(f"    提取到文本，分割成 {len(chunks)} 个块。")

        for i, chunk in enumerate(chunks):
            chunk_count += 1
            # 按字节截断，确保不超过 Milvus VARCHAR 的字节限制
            truncated_chunk = truncate_by_bytes(chunk, max_bytes, encoding='utf-8')
            if len(truncated_chunk.encode('utf-8')) > max_bytes:
                # 理论上不会发生，但作为保险
                truncated_chunk = truncated_chunk.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')
            if truncated_chunk != chunk:
                truncated_examples += 1

            milvus_entry = {
                "id": f"{filename}_{i}",
                "title": title or filename,
                "abstract": truncated_chunk,  # 已按字节截断到允许长度
                "source_file": filename,
                "chunk_index": i
            }
            all_data_for_milvus.append(milvus_entry)
    else:
        print(f"    警告：未能从 {filename} 提取有效文本内容。")

print(f"\n处理完成。共处理 {file_count} 个文件，生成 {chunk_count} 个文本块。截断示例数: {truncated_examples}")

try:
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_data_for_milvus, f, ensure_ascii=False, indent=4)
    print(f"结果已保存到: {output_json_path}")
except Exception as e:
    print(f"错误：无法写入 JSON 文件 {output_json_path}: {e}")
