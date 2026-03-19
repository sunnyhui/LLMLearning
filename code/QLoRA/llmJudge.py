import json
import os
from openai import OpenAI

# 初始化客户端，指向 DeepSeek API
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "your_deepseek_api_key_here"),
    base_url="https://api.deepseek.com/v1"
)

def evaluate_with_llm_judge(instruction: str, prediction: str, reference: str) -> dict:
    """
    使用 LLM 作为裁判，对微调模型的输出进行多维度打分
    """
    # 构建评分量表 (Rubric) Prompt
    system_prompt = """你是一个严谨的算法评估专家。你需要根据提供的【用户指令】和【参考答案】，对【模型输出】进行多维度评分。
请从以下三个维度打分（1-5分，5分为最优）：
1. 准确性 (Accuracy)：模型输出是否与参考答案的核心事实或语义一致？
2. 完整性 (Completeness)：模型是否回答了指令要求的所有关键点？
3. 连贯性 (Coherence)：文本表达是否自然、专业、符合逻辑？

请务必严格按照以下 JSON 格式输出你的评估结果，不要包含任何其他分析文字：
{"accuracy": 0, "completeness": 0, "coherence": 0, "reasoning": "简要的一句话打分理由"}
"""

    user_prompt = f"""
【用户指令】: {instruction}
【参考答案】: {reference}
【模型输出】: {prediction}
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # 强制输出 JSON 格式
            response_format={"type": "json_object"}, 
            temperature=0.1 # 评估任务需要低温度以保证输出稳定性
        )
        
        # 解析返回的 JSON 字符串
        result_str = response.choices[0].message.content
        return json.loads(result_str)
        
    except Exception as e:
        print(f"API 调用或解析失败: {e}")
        return {"accuracy": 0, "completeness": 0, "coherence": 0, "reasoning": "Error"}

# --- 测试用例 ---
if __name__ == "__main__":
    inst = "请解释一下什么是神经声码器中的 HiFi-GAN 架构？"
    ref = "HiFi-GAN 是一种基于生成对抗网络的高保真神经声码器，它主要由一个生成器和多个判别器（如多尺度和多周期判别器）组成，能够高效地将声学特征转换为高质量的音频波形。"
    pred = "HiFi-GAN 是个声码器，用 GAN 做的，包含生成器和判别器，用来把特征变成声音，速度挺快的。"
    
    # 注意：运行此代码需要替换真实的 API Key
    # evaluation = evaluate_with_llm_judge(inst, pred, ref)
    # print("LLM 裁判打分结果:", evaluation)