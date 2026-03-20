import json

with open(r'c:\Users\liushihui\Desktop\LLM\code\rag\testset\qieshitianxia\output\chunks.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

chunks = data['chunks']
sizes = [len(c['content']) for c in chunks]

print('='*60)
print('使用 RecursiveCharacterTextSplitter 分割统计')
print('='*60)
print(f'总章节数: {data["metadata"]["total_chapters"]}')
print(f'总文本块数: {data["metadata"]["total_chunks"]}')
print(f'块大小: {data["metadata"]["split_config"]["chunk_size"]}')
print(f'块重叠: {data["metadata"]["split_config"]["chunk_overlap"]}')
print()
print(f'文本块大小统计:')
print(f'  平均大小: {sum(sizes)/len(sizes):.1f} 字符')
print(f'  最小: {min(sizes)} 字符')
print(f'  最大: {max(sizes)} 字符')
print(f'  中位数: {sorted(sizes)[len(sizes)//2]} 字符')
print()
print(f'总字符数: {sum(sizes)}')
print('='*60)
