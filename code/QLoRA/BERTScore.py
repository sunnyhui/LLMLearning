import evaluate
from typing import List, Dict

# 加载 BERTScore 评测指标
# 首次运行会自动下载底层模型，推荐使用多语言版的 RoBERTa
bertscore = evaluate.load("bertscore")

def automated_screening(predictions: List[str], references: List[str], required_keywords: List[str] = None) -> Dict:
    """
    自动化初筛：计算 BERTScore 并校验关键词命中率
    """
    print("正在计算 BERTScore...")
    # lang="zh" 会默认使用 bert-base-chinese，你也可以指定 model_type
    results = bertscore.compute(predictions=predictions, references=references, lang="zh")
    
    # 提取 F1 分数
    f1_scores = results['f1']
    avg_f1 = sum(f1_scores) / len(f1_scores)
    
    screening_report = {
        "average_f1": avg_f1,
        "details": []
    }
    
    # 遍历每个样本，结合关键词硬约束进行评估
    for i, (pred, ref, f1) in enumerate(zip(predictions, references, f1_scores)):
        keyword_passed = True
        missing_keys = []
        
        if required_keywords:
            for kw in required_keywords:
                if kw not in pred:
                    keyword_passed = False
                    missing_keys.append(kw)
                    
        screening_report["details"].append({
            "sample_index": i,
            "bertscore_f1": f1,
            "keyword_passed": keyword_passed,
            "missing_keywords": missing_keys,
            "prediction_snippet": pred[:50] + "..." # 截断展示
        })
        
    return screening_report

# --- 测试用例 ---
if __name__ == "__main__":
    preds = ["该算法通过提取梅尔频谱特征来优化声音的质感。", "这个模型效果很好，能直接输出音频。"]
    refs = ["该方法利用梅尔频谱图特征，显著提升了合成语音的音色质感。", "该模型具有优异的表现，支持端到端的音频生成。"]
    # keywords = ["特征", "模型"]
    
    # report = automated_screening(preds, refs, required_keywords=keywords)
    report = automated_screening(preds, refs)
    print(f"平均 BERTScore F1: {report['average_f1']:.4f}")
    print(f"首个样本详情: {report['details'][0]}")