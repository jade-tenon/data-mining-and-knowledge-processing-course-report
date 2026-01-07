import torch
from transformers import AutoTokenizer
from config import Config
from model import SentimentClassifier
import sys

# 加载配置
config = Config()

# 加载分词器和模型
print("加载分词器和模型...")
tokenizer = AutoTokenizer.from_pretrained(config.model_name, local_files_only=True)
model = SentimentClassifier(config.model_name, config.num_classes)
model.load_model(config.model_save_path)
model.eval()

def predict_sentiment(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=config.max_seq_length,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs[0][pred].item()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = input("请输入一句话进行情��判断：")
    pred, conf = predict_sentiment(text)
    label_map = {0: "负面", 1: "正面"}
    print(f"预测结果: {label_map.get(pred, pred)} (置信度: {conf:.4f})")

