import sys
from gensim.models import Word2Vec

def load_model(model_path):
    """加载训练好的Word2Vec模型"""
    model = Word2Vec.load(model_path)
    return model

def compute_similarity(model, word1, word2):
    """计算两个单词的词向量余弦相似度"""
    if (word1 in model.wv) and (word2 in model.wv):
        return model.wv.similarity(word1, word2)
    else:
        missing = []
        if word1 not in model.wv:
            missing.append(word1)
        if word2 not in model.wv:
            missing.append(word2)
        print(f"词表中缺失: {', '.join(missing)}")
        return None

def find_most_similar(model, word, topn=5):
    """查找与指定单词最相似的topn个词"""
    if word in model.wv:
        return model.wv.most_similar(word, topn=topn)
    else:
        print(f"词表中缺失: {word}")
        return None

def main():
    if len(sys.argv) < 2:
        print("用法: python infer_word_similarity.py word2vec_sentiment.model")
        sys.exit(1)

    model_path = sys.argv[1]
    model = load_model(model_path)

    print("Word2Vec模型加载完毕。")

    while True:
        print("\n选项：")
        print("1. 计算两个词的余弦相似度")
        print("2. 查找最相似的词")
        print("3. 退出")
        choice = input("请选择功能（1/2/3）：").strip()
        if choice == '1':
            w1 = input("请输入第一个词：").strip().lower()
            w2 = input("请输入第二个词：").strip().lower()
            sim = compute_similarity(model, w1, w2)
            if sim is not None:
                print(f"“{w1}”与“{w2}”的相似度为: {sim:.4f}")
        elif choice == '2':
            w = input("请输入要查询的词：").strip().lower()
            res = find_most_similar(model, w)
            if res is not None:
                print(f"与“{w}”最相似的词：")
                for other, score in res:
                    print(f"  {other}: {score:.4f}")
        elif choice == '3':
            print("退出。")
            break
        else:
            print("选择无效，请输入1、2或3。")

if __name__ == "__main__":
    main()