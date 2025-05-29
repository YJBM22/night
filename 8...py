from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 设置国内镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载模型和分词器
model_name = "IDEA-CCNL/Wenzhong-GPT2-110M"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# 显式设置pad_token（使用eos_token）
tokenizer.pad_token = tokenizer.eos_token  # 添加这一行

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    use_safetensors=True
)

# 生成配置优化
generation_config = {
    "max_new_tokens": 150,        # 生成最大长度
    "temperature": 0.7,           # 控制随机性（0.1~1.0）
    "top_p": 0.9,                # 核心采样比例
    "repetition_penalty": 1.2,    # 抑制重复
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id  # 显式指定填充标记
}

def generate_continuation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 执行生成
print("=== 续写结果 ===")
print(generate_continuation("当人类第一次踏上火星"))