from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_name = "bert-base-chinese-finetuned-sst"
try:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
except:

    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


label_map = {0: "负面", 1: "正面"}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    return label_map[torch.argmax(probs).item()]


test_cases = [
    ("剧情设定新颖不落俗套，每个转折都让人惊喜。", "正面"),
    ("汤汁洒得到处都是，包装太随便了。", "负面")

]

print("=== 测试结果 ===")
for text, expected in test_cases:
    result = predict_sentiment(text)
    color = '\033[92m' if result == expected else '\033[91m'
    print(f"输入：{text}")
    print(f"预测：{color}{result}\033[0m (预期：{expected})")
    print("-"*50)