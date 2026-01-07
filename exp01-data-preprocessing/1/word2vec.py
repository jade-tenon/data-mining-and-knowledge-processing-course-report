import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from nltk.tokenize import word_tokenize
import nltk
import re
import matplotlib.pyplot as plt
import os

nltk.download('punkt')

class LossLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.losses = []
        self.previous_loss = 0.0
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        current_loss = loss - self.previous_loss
        self.losses.append(current_loss)
        self.previous_loss = loss
        print(f'Epoch {self.epoch} | Loss: {current_loss:.2f}')
        self.epoch += 1

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    return tokens

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    if df.shape[1] < 3:
        raise ValueError("CSV需要包含标签、标题、评论三列")
    text_col = (df.iloc[:,1] + " " + df.iloc[:,2]).astype(str)
    corpus = [preprocess_text(text) for text in text_col]
    labels = df.iloc[:,0].values
    return corpus, labels, text_col

def build_doc_vectors(all_text_tokens, model):
    doc_vectors = np.array([
        get_document_vector(tokens, model)
        for tokens in all_text_tokens
    ])
    return doc_vectors

def get_document_vector(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def plot_loss_curve(losses, out_path='loss_curve.png'):
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Word2Vec Training Loss per Epoch')
    plt.grid()
    plt.savefig(out_path)
    print(f'Loss curve saved to {out_path}')
    # plt.show()   # 注释避免阻塞

def main():
    data_path = 'dataset/train2.csv'
    model_path = "word2vec_sentiment.model"
    vector_size = 100
    window = 5
    min_count = 1
    workers = 1       # 多线程有时在Windows会有bug,debug建议为1
    epochs = 5        # 每次增量继续训练的轮数

    print("开始加载数据")
    corpus, labels, text_col = load_and_preprocess_data(data_path)
    print("数据加载完成，样本数:", len(corpus))
    loss_logger = LossLogger()

    if os.path.exists(model_path):
        print("发现已有的Word2Vec模型，载入继续训练……")
        model = Word2Vec.load(model_path)
        # 增量训练需update词表
        print("更新词表……")
        model.build_vocab(corpus, update=True)
        print("继续训练中……")
        model.train(corpus, total_examples=len(corpus), epochs=epochs, callbacks=[loss_logger])
    else:
        print("未找到模型，重新训练新模型。")
        model = Word2Vec(
            sentences=corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            compute_loss=True,
            epochs=epochs,
            callbacks=[loss_logger]
        )
    print("训练结束，保存模型。")
    model.save(model_path)
    print("开始提取文档向量")
    X = build_doc_vectors(corpus, model)
    y = labels

    print("文档向量形状:", X.shape)
    print("标签形状:", y.shape)
    plot_loss_curve(loss_logger.losses)
    word = "great"
    if word in model.wv.key_to_index:
        similar_words = model.wv.most_similar(word, topn=5)
        print(f"\n与'{word}'最相似的词:")
        for word_, score in similar_words:
            print(f"{word_}: {score:.4f}")

if __name__ == "__main__":
    main()