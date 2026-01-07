import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import torch

class SentimentClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(SentimentClassifier, self).__init__()
        # 使用Qwen2专用模型加载
        self.qwen = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True, num_labels=num_classes)
    def forward(self, input_ids, attention_mask):
        outputs = self.qwen(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    def save_model(self, path):
        torch.save(self.qwen.state_dict(), path)
    def load_model(self, path):
        self.qwen.load_state_dict(torch.load(path))
