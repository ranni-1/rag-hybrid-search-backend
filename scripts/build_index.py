import os
import json

DATA_DIR = '../data'


def parse_markdown_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 简单粗暴的切分：按二级标题 "## " 切分
    sections = content.split('\n## ')
    chunks = []

    # 获取文件名作为基础ID
    filename = os.path.basename(filepath).replace('.md', '')

    for i, section in enumerate(sections):
        if not section.strip():
            continue

        # 如果是第一段，可能没有 "## " 前缀
        if i == 0:
            chunk_id = f"{filename}_intro"
            text = section.strip()
        else:
            # 把切掉的 "## " 加回去，作为内容的开头
            chunk_id = f"{filename}_sec_{i:02d}"
            text = "## " + section.strip()

        chunks.append({
            "chunk_id": chunk_id,
            "text": text
        })

    return chunks


def main():
    if not os.path.exists(DATA_DIR):
        print(f"错误: 找不到 {DATA_DIR} 文件夹，请先建好文件夹并放入 md 文件！")
        return

    all_chunks = []
    # 遍历 data 文件夹下所有 .md 文件
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.md'):
            filepath = os.path.join(DATA_DIR, filename)
            file_chunks = parse_markdown_file(filepath)
            all_chunks.extend(file_chunks)
            print(f"已处理: {filename}, 切分出 {len(file_chunks)} 个 chunk")

    # 把切好的结果保存到 chunks.jsonl 文件里，供后面建索引使用
    with open('../outputs/chunks.jsonl', 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f"\n处理完成！总共生成了 {len(all_chunks)} 个 chunks。")
    print("结果已保存至 chunks.jsonl")


if __name__ == '__main__':
    main()