from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型和分词器
model_name = "IDEA-CCNL/Wenzhong-GPT2-110M"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # 显式设置pad_token（使用eos_token）
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
except Exception as e:
    print(f"加载模型失败: {e}")
    print("尝试使用更小的模型...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

# 生成配置优化
generation_config = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_p": 0.9,
    "repetition_penalty": 1.2,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id
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