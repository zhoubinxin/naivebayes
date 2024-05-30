import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

import naiveBayes as nb


def compute_mutual_information(X):
    """
    计算数据集中每对特征之间的互信息，并构建互信息矩阵。
    :param X:
    :return:
    """
    n_features = X.shape[1]  # 特征数量
    mi_matrix = np.zeros((n_features, n_features))

    # 计算每对特征之间的互信息
    with tqdm(total=n_features * (n_features - 1) // 2, desc="计算互信息") as pbar:
        for i in range(n_features):
            for j in range(i + 1, n_features):
                mi_matrix[i, j] = mutual_information(X[:, i], X[:, j])
                mi_matrix[j, i] = mi_matrix[i, j]

                pbar.update(1)

    return mi_matrix


# 计算两个特征之间的互信息
def mutual_information(x, y):
    p_xy = pd.crosstab(x, y, normalize=True)  # 计算联合概率
    p_x = p_xy.sum(axis=1)  # 计算边际概率P(x)
    p_y = p_xy.sum(axis=0)  # 计算边际概率P(y)
    mi = 0.0
    for i in p_xy.index:
        for j in p_xy.columns:
            if p_xy.at[i, j] > 0:
                mi += p_xy.at[i, j] * np.log(p_xy.at[i, j] / (p_x[i] * p_y[j]))  # 计算互信息
    return mi


# 使用Prim算法构建最大权重生成树
def prim_algorithm(mi_matrix, vocabList):
    n_features = mi_matrix.shape[0]
    selected_nodes = [0]  # 初始化包含第一个节点
    edges = []

    # 进度条
    with tqdm(total=n_features - 1, desc="构建树") as pbar:
        while len(selected_nodes) < n_features:
            max_weight = -np.inf  # 初始化最大权重为负无穷
            new_edge = None  # 用于存储找到的新的边
            for i in selected_nodes:
                for j in range(n_features):
                    if j not in selected_nodes and mi_matrix[i, j] > max_weight:  # 如果j未被选中且权重大于当前最大权重
                        max_weight = mi_matrix[i, j]
                        new_edge = (i, j)
            edges.append(new_edge)
            selected_nodes.append(new_edge[1])
            pbar.update(1)

    # 创建networkx图
    G = nx.Graph()
    G.add_edges_from(edges)

    # 为每个节点设置标签
    labels = {i: vocabList[i] for i in range(len(vocabList))}

    # 绘制图
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)  # 布局
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=500, node_color="lightblue", font_size=10,
            font_weight="bold", edge_color="gray")
    plt.title("最大权重生成树")
    plt.show()

    return edges


class TAN:
    def __init__(self, vocabList):
        self.class_prior = {}  # 存储类的先验概率
        self.feature_probs = {}  # 存储特征的条件概率
        self.edges = []  # 存储树的边
        self.vocabList = vocabList

    def fit(self, X, y):
        n_samples, n_features = X.shape  # 获取样本数和特征数
        self.classes, counts = np.unique(y, return_counts=True)  # 获取类标签及其计数
        self.class_prior = dict(zip(self.classes, counts / n_samples))  # 计算先验概率

        mi_matrix = compute_mutual_information(X)  # 计算互信息矩阵
        self.edges = prim_algorithm(mi_matrix, self.vocabList)  # 构建最大权重生成树

        self.feature_probs = {c: [{} for _ in range(n_features)] for c in self.classes}  # 初始化条件概率
        for c in tqdm(self.classes, desc="计算条件概率"):
            X_c = X[y == c]  # 获取属于类c的样本
            X_c_df = pd.DataFrame(X_c)
            for i in range(n_features):
                parent = next((edge[0] for edge in self.edges if edge[1] == i), None)  # 找到特征i的父节点
                if parent is None:
                    probs = X_c_df[i].value_counts(normalize=True).to_dict()  # 计算P(X_i|C)
                else:
                    probs = X_c_df.groupby(parent)[i].value_counts(normalize=True).to_dict()  # 计算P(X_i|X_parent, C)
                self.feature_probs[c][i] = probs

    def predict(self, X):
        n_samples, n_features = X.shape
        log_prob = np.zeros((n_samples, len(self.classes)))

        for i, c in enumerate(self.classes):
            log_prob[:, i] += np.log(self.class_prior[c])  # 加上类的先验概率
            for j in range(n_features):
                parent = next((edge[0] for edge in self.edges if edge[1] == j), None)  # 找到特征j的父节点
                if parent is None:
                    probs = np.array([self.feature_probs[c][j].get(x, 1e-6) for x in X[:, j]])  # 计算P(X_j|C)
                else:
                    probs = np.array([self.feature_probs[c][j].get((X[k, parent], X[k, j]), 1e-6) for k in
                                      range(n_samples)])  # 计算P(X_j|X_parent, C)
                log_prob[:, i] += np.log(probs)

        return self.classes[np.argmax(log_prob, axis=1)]  # 返回概率最大的类


# 示例用法
if __name__ == "__main__":
    # 加载数据集
    docs, label = nb.loadTestDataSet()
    # 创建词汇表
    vocabList = nb.createVocabList(docs)

    # 构建词向量矩阵
    trainMat = []
    for inputSet in tqdm(docs, desc='构建词向量矩阵'):
        trainMat.append(nb.setOfWords2Vec(vocabList, inputSet))
        # trainMat.append(nb.bagOfWords2VecMN(vocabList, inputSet))
    # tfidf = nb.TFIDF(docs, vocabList)
    # trainMat = tfidf.calc_tfidf()

    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(trainMat, label, test_size=0.2, random_state=42)

    # 训练模型
    model = TAN(vocabList)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{conf_matrix}\n")
