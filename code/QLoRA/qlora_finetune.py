#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QLoRA微调Qwen3.5模型使用Huatuo-26M医疗数据集
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging
)
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

# 配置日志
logging.set_verbosity_info()

# 全局配置
class Config:
    # 模型配置
    model_name = "Qwen/Qwen3.5-1.8B"
    # 数据集配置
    dataset_name = "FreedomIntelligence/Huatuo26M-Lite"  # 使用精简版数据集
    # 训练配置
    output_dir = "./output/qwen3.5-huatuo"
    per_device_train_batch_size = 4
    per_device_eval_batch_size = 4
    gradient_accumulation_steps = 4
    learning_rate = 2e-4
    max_seq_length = 1024
    num_train_epochs = 3
    logging_steps = 100
    save_steps = 1000
    save_total_limit = 3
    
    # QLoRA配置
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# 全局测试配置
TEST_SAMPLES = [
    "感冒了应该吃什么药？",
    "头痛怎么办？",
    "如何预防高血压？"
]

# 提示模板
PROMPT_TEMPLATE = """用户现在需要医疗咨询。请作为专业医生回答以下问题。

问题：{question}

医生回答："""


# 加载数据集
def load_huatuo_dataset():
    """加载并预处理Huatuo-26M-Lite数据集"""
    print("加载Huatuo-26M-Lite数据集...")
    dataset = load_dataset(Config.dataset_name)
    
    # 查看数据集结构
    print(f"数据集结构: {dataset}")
    print(f"训练集大小: {len(dataset['train'])}")
    print(f"测试集大小: {len(dataset['test'])}")
    
    # 查看样例数据
    print("\n样例数据:")
    for i in range(3):
        sample = dataset['train'][i]
        print(f"问题: {sample['Question']}")
        print(f"答案: {sample['Answer']}")
        if 'Department' in sample:
            print(f"科室: {sample['Department']}")
        if 'Disease' in sample:
            print(f"疾病: {sample['Disease']}")
        print("-" * 50)
    
    return dataset

# 数据预处理
def preprocess_function(examples, tokenizer):
    """预处理数据，转换为模型输入格式"""
    # 构建对话格式
    prompts = []
    for question, answer in zip(examples['Question'], examples['Answer']):
        # Qwen的对话格式
        prompt = f"""用户现在需要医疗咨询。请作为专业医生回答以下问题。

问题：{question}

医生回答：{answer}"""
        prompts.append(prompt)
    
    # 编码
    inputs = tokenizer(
        prompts,
        max_length=Config.max_seq_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # 设置labels
    inputs["labels"] = inputs["input_ids"].clone()
    
    return inputs

# 加载模型和分词器
def load_model_and_tokenizer():
    """加载模型和分词器，配置QLoRA"""
    print(f"加载模型: {Config.model_name}")
    
    # 配置bitsandbytes
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        Config.model_name,
        trust_remote_code=True
    )
    
    # 配置分词器
    tokenizer.pad_token = tokenizer.eos_token
    
    # 准备模型进行kbit训练
    model = prepare_model_for_kbit_training(model)
    
    # 配置LoRA
    peft_config = LoraConfig(
        r=Config.lora_r,
        lora_alpha=Config.lora_alpha,
        target_modules=Config.lora_target_modules,
        lora_dropout=Config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 获取PEFT模型
    model = get_peft_model(model, peft_config)
    
    # 打印模型信息
    model.print_trainable_parameters()
    
    return model, tokenizer

# 训练函数
def train():
    """执行训练"""
    # 创建输出目录
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # 加载数据集
    dataset = load_huatuo_dataset()
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=Config.output_dir,
        per_device_train_batch_size=Config.per_device_train_batch_size,
        per_device_eval_batch_size=Config.per_device_eval_batch_size,
        gradient_accumulation_steps=Config.gradient_accumulation_steps,
        learning_rate=Config.learning_rate,
        num_train_epochs=Config.num_train_epochs,
        logging_steps=Config.logging_steps,
        save_steps=Config.save_steps,
        save_total_limit=Config.save_total_limit,
        evaluation_strategy="steps",
        eval_steps=Config.logging_steps,
        fp16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        report_to="tensorboard"
    )
    
    # 创建SFT训练器
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        args=training_args,
        packing=False,
        max_seq_length=Config.max_seq_length
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存模型
    print("保存模型...")
    trainer.save_model()
    
    # 保存分词器
    tokenizer.save_pretrained(Config.output_dir)
    
    print("训练完成！")

# 测试原始模型函数
def test_original_model():
    """测试未进行微调的原始模型"""
    print("\n加载原始模型...")
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ),
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        Config.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 创建推理管道
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    print("\n原始模型测试结果:")
    for sample in TEST_SAMPLES:
        prompt = PROMPT_TEMPLATE.format(question=sample)
        
        result = pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        print(f"问题: {sample}")
        print(f"回答: {result[0]['generated_text'].split('医生回答：')[1].strip()}")
        print("-" * 50)

# 测试函数
def test():
    """测试微调后的模型"""
    print("\n加载微调后的模型...")
    
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.model_name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        ),
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        Config.model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载PEFT模型
    model = PeftModel.from_pretrained(base_model, Config.output_dir)
    
    # 创建推理管道
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    
    print("\n微调后模型测试结果:")
    for sample in TEST_SAMPLES:
        prompt = PROMPT_TEMPLATE.format(question=sample)
        
        result = pipe(
            prompt,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )
        
        print(f"问题: {sample}")
        print(f"回答: {result[0]['generated_text'].split('医生回答：')[1].strip()}")
        print("-" * 50)
        


if __name__ == "__main__":
    # 先测试原始模型
    test_original_model()
    
    # 执行训练
    train()
    
    # 执行测试
    test()

