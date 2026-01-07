import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class JobNode2VecNet(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, hidden_dims=[256, 128], dropout=0.3):
        super(JobNode2VecNet, self).__init__()

        # 编码器网络
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # 最终的嵌入层
        self.embedding = nn.Linear(prev_dim, embedding_dim)

        # 解码器（用于自监督学习）
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        embeddings = self.embedding(encoded)
        reconstructed = self.decoder(embeddings)
        return embeddings, reconstructed


class GraphBuilder:
    def __init__(self, data, similarity_threshold=0.7):
        self.data = data
        self.similarity_threshold = similarity_threshold
        self.graph = nx.Graph()

    def preprocess_data(self):
        """预处理数据，处理数值和分类特征"""
        df = self.data.copy()

        # 处理数值特征
        numerical_features = ['max_salary', 'med_salary', 'min_salary', 'views',
                              'applies', 'original_listed_time', 'remote_allowed',
                              'expiry', 'closed_time', 'listed_time', 'normalized_salary',
                              'fips']

        # 处理缺失值
        for col in numerical_features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # 标准化数值特征
        scaler = StandardScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])

        # 处理分类特征
        categorical_features = ['pay_period', 'formatted_work_type', 'application_type',
                                'formatted_experience_level', 'work_type', 'currency',
                                'compensation_type', 'location']

        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                # 归一化分类特征
                df[col] = df[col] / df[col].max()

        # 处理文本特征（简单处理）
        text_features = ['title', 'description', 'skills_desc']
        for col in text_features:
            if col in df.columns:
                # 使用简单的长度特征
                df[f'{col}_len'] = df[col].astype(str).str.len()
                df[f'{col}_len'] = df[f'{col}_len'] / df[f'{col}_len'].max()

        # 选择用于相似度计算的数值列
        feature_columns = numerical_features + categorical_features + [f'{col}_len' for col in text_features]
        feature_columns = [col for col in feature_columns if col in df.columns]

        return df, feature_columns

    def build_similarity_graph(self):
        """基于特征相似度构建图"""
        df, feature_columns = self.preprocess_data()

        # 计算相似度矩阵
        features = df[feature_columns].values
        similarity_matrix = cosine_similarity(features)

        # 添加节点
        for i in range(len(df)):
            self.graph.add_node(i, **df.iloc[i].to_dict())

        # 添加边（基于相似度）
        for i in range(len(df)):
            for j in range(i + 1, len(df)):
                if similarity_matrix[i, j] > self.similarity_threshold:
                    self.graph.add_edge(i, j, weight=similarity_matrix[i, j])

        print(f"构建的图包含 {self.graph.number_of_nodes()} 个节点和 {self.graph.number_of_edges()} 条边")
        return self.graph


class Node2VecTrainer:
    def __init__(self, graph, embedding_dim=128, walk_length=30, num_walks=200):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks

    def train_node2vec(self):
        """训练node2vec模型"""
        # 初始化node2vec
        node2vec = Node2Vec(
            self.graph,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=4
        )

        # 训练模型
        model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # 获取嵌入向量
        embeddings = {}
        for node in self.graph.nodes():
            embeddings[node] = model.wv[str(node)]

        return embeddings, model


class NeuralNetworkTrainer:
    def __init__(self, input_dim, embeddings):
        self.input_dim = input_dim
        self.embeddings = embeddings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data(self):
        """准备训练数据"""
        X = []
        y = []

        for node, embedding in self.embeddings.items():
            # 使用原始特征作为输入，node2vec嵌入作为目标
            X.append(list(self.embeddings[node]))
            y.append(list(self.embeddings[node]))

        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y)

        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        return train_loader

    def train_model(self, num_epochs=100):
        """训练神经网络"""
        model = JobNode2VecNet(
            input_dim=self.input_dim,
            embedding_dim=self.input_dim  # 输入输出维度相同
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        train_loader = self.prepare_data()

        losses = []
        model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                embeddings, reconstructed = model(batch_X)
                loss = criterion(reconstructed, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

        return model, losses


def main():
    # 加载数据（请替换为您的实际数据路径）
    # data = pd.read_csv('your_dataset.csv')

    # 由于没有实际数据，这里创建一个示例数据集
    print("创建示例数据集...")
    n_samples = 756
    n_features = 31

    # 创建模拟数据
    feature_names = [
        'job_id', 'company_name', 'title', 'description', 'max_salary', 'pay_period',
        'location', 'company_id', 'views', 'med_salary', 'min_salary', 'formatted_work_type',
        'applies', 'original_listed_time', 'remote_allowed', 'job_posting_url',
        'application_url', 'application_type', 'expiry', 'closed_time',
        'formatted_experience_level', 'skills_desc', 'listed_time', 'posting_domain',
        'sponsored', 'work_type', 'currency', 'compensation_type', 'normalized_salary',
        'zip_code', 'fips'
    ]

    # 创建模拟数据
    np.random.seed(42)
    data = pd.DataFrame(np.random.randn(n_samples, n_features), columns=feature_names)

    # 添加一些类别特征
    data['pay_period'] = np.random.choice(['hourly', 'monthly', 'yearly'], n_samples)
    data['formatted_work_type'] = np.random.choice(['full_time', 'part_time', 'contract'], n_samples)
    data['location'] = np.random.choice(['New York', 'San Francisco', 'Remote'], n_samples)

    print(f"数据集形状: {data.shape}")

    # 1. 构建图
    print("\n1. 构建相似度图...")
    graph_builder = GraphBuilder(data, similarity_threshold=0.6)
    graph = graph_builder.build_similarity_graph()

    # 2. 训练node2vec
    print("\n2. 训练Node2Vec模型...")
    node2vec_trainer = Node2VecTrainer(graph, embedding_dim=128)
    embeddings, node2vec_model = node2vec_trainer.train_node2vec()

    # 3. 训练神经网络
    print("\n3. 训练神经网络...")
    nn_trainer = NeuralNetworkTrainer(input_dim=128, embeddings=embeddings)
    model, losses = nn_trainer.train_model(num_epochs=100)

    # 4. 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

    # 5. 保存模型和嵌入
    torch.save(model.state_dict(), 'job_node2vec_model.pth')

    # 保存嵌入向量
    embedding_df = pd.DataFrame.from_dict(embeddings, orient='index')
    embedding_df.to_csv('job_embeddings.csv')

    print("\n训练完成！")
    print(f"最终嵌入维度: {embedding_df.shape}")
    print("模型已保存为 'job_node2vec_model.pth'")
    print("嵌入向量已保存为 'job_embeddings.csv'")


def analyze_embeddings(embeddings, data, top_k=10):
    """分析学习到的嵌入"""
    print("\n分析嵌入向量...")

    # 转换为numpy数组
    embedding_matrix = np.array(list(embeddings.values()))

    # 计算相似工作
    similarity_matrix = cosine_similarity(embedding_matrix)

    # 为每个工作找到最相似的工作
    similar_jobs = {}
    for i in range(len(embedding_matrix)):
        similarities = similarity_matrix[i]
        # 排除自身
        similarities[i] = -1
        most_similar_indices = np.argsort(similarities)[-top_k:][::-1]
        similar_jobs[i] = most_similar_indices

    return similar_jobs


if __name__ == "__main__":
    main()