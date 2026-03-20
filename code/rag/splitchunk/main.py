import json
import os
from typing import List, Dict, Any
from dataclasses import asdict
from splitter import ChapterSplitter, ChunkSplitter, MetadataExtractor
from splitter.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def save_output(data: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_document(input_file: str, output_file: str) -> Dict[str, Any]:
    print(f"正在读取文件: {input_file}")
    text = load_text(input_file)
    print(f"文件大小: {len(text)} 字符")
    
    chapter_splitter = ChapterSplitter()
    chunk_splitter = ChunkSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    metadata_extractor = MetadataExtractor()
    
    print("\n开始章节分割...")
    chapters = chapter_splitter.split(text)
    print(f"识别到 {len(chapters)} 个章节")
    
    all_chunks = []
    chunk_id_counter = 0
    
    print("\n开始文本块分割和元数据提取...")
    for chapter in chapters:
        print(f"  处理章节: {chapter.title}")
        
        chunks_with_metadata = chunk_splitter.split(chapter.content)
        
        for chunk_idx, chunk in enumerate(chunks_with_metadata):
            chunk_content = chunk['content']
            metadata = metadata_extractor.extract(
                chunk_content=chunk_content,
                chapter_title=chapter.title,
                chapter_index=chapter.index,
                chunk_index=chunk_idx
            )
            
            chunk_data = {
                "id": f"chunk_{chunk_id_counter:04d}",
                "content": chunk_content,
                "metadata": metadata
            }
            
            all_chunks.append(chunk_data)
            chunk_id_counter += 1
    
    output_data = {
        "metadata": {
            "source_file": "且试天下.txt",
            "total_chapters": len(chapters),
            "total_chunks": len(all_chunks),
            "split_config": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP
            }
        },
        "chunks": all_chunks
    }
    
    print(f"\n保存结果到: {output_file}")
    save_output(output_data, output_file)
    
    return output_data


def print_statistics(result: Dict[str, Any]) -> None:
    print("\n" + "="*60)
    print("分割统计信息")
    print("="*60)
    print(f"总章节数: {result['metadata']['total_chapters']}")
    print(f"总文本块数: {result['metadata']['total_chunks']}")
    print(f"块大小: {result['metadata']['split_config']['chunk_size']}")
    print(f"块重叠: {result['metadata']['split_config']['chunk_overlap']}")
    
    chunks = result['chunks']
    if chunks:
        sizes = [len(c['content']) for c in chunks]
        avg_size = sum(sizes) / len(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        
        print(f"\n文本块大小统计:")
        print(f"  平均大小: {avg_size:.1f} 字符")
        print(f"  最小: {min_size} 字符")
        print(f"  最大: {max_size} 字符")
        
        total_characters = sum(sizes)
        print(f"\n总字符数: {total_characters}")
    
    print("="*60)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = r'C:\Users\liushihui\Desktop\LLM\code\rag\testset\qieshitianxia\content.txt'
    output_file = os.path.join(base_dir, "output", "chunks1.json")
    
    result = process_document(input_file, output_file)
    print_statistics(result)
    
    print(f"\n处理完成! 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
