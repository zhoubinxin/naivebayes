from naiveBayes import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split


class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in tqdm(range(self.n_iters), desc='训练SVM'):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)

    def decision_function(self, X):
        return np.dot(X, self.w) - self.b


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def tsvm_nb_algorithm(X, y):
    # 初始化距离矩阵
    n_samples = len(X)
    distance_matrix = np.full((n_samples, n_samples), np.inf)

    for i in tqdm(range(n_samples),desc='计算距离矩阵'):
        for j in range(n_samples):
            if i != j:
                distance_matrix[i, j] = euclidean_distance(X[i], X[j])

    # 初始化每个点的最近邻和最短距离
    nearest_neighbors = np.zeros(n_samples, dtype=int)
    min_distances = np.full(n_samples, np.inf)

    for i in tqdm(range(n_samples),desc='选择最近邻'):
        for j in range(n_samples):
            if distance_matrix[i, j] < min_distances[i]:
                min_distances[i] = distance_matrix[i, j]
                nearest_neighbors[i] = j

    # 初始化标志矩阵
    flags = np.ones(n_samples)

    for i in tqdm(range(n_samples),desc='计算标志矩阵'):
        neighbor_idx = nearest_neighbors[i]
        if y[i] != y[neighbor_idx]:
            flags[i] = -1
        else:
            flags[i] = 1

    # 修剪样本集
    for i in tqdm(range(n_samples),desc='修剪样本集'):
        neighbor_idx = nearest_neighbors[i]
        if flags[i] == -1:
            # 选择删除点，优先删除距离较远的点
            if min_distances[i] < min_distances[neighbor_idx]:
                X = np.delete(X, i, axis=0)
                y = np.delete(y, i, axis=0)
            else:
                X = np.delete(X, neighbor_idx, axis=0)
                y = np.delete(y, neighbor_idx, axis=0)

    # 再次用NB算法训练
    nb = NaiveBayes()
    nb.fit(X, y)

    return nb


def main():
    # 加载数据集
    docs, label = loadDataSet()
    # 创建词汇表
    vocabList = createVocabList(docs)

    # 构建词向量矩阵
    tfidf = TFIDF(docs, vocabList)
    trainMat = tfidf.calc_tfidf()

    trainMat = np.array(trainMat)
    label = np.array(label)
    X_train, X_test, y_train, y_test = train_test_split(trainMat, label, test_size=0.2, random_state=1)

    # 初始训练
    nb_initial = NaiveBayes()
    nb_initial.fit(X_train, y_train)
    initial_predictions = nb_initial.predict(X_train)

    # 构建最优分类超平面
    svm = LinearSVM()
    svm.fit(X_train, initial_predictions)

    distances = svm.decision_function(X_train)
    threshold = 0.2
    selected_samples = np.abs(distances) > threshold

    X_optimized = X_train[selected_samples]
    y_optimized = y_train[selected_samples]

    nb = tsvm_nb_algorithm(X_optimized, y_optimized)
    y_pred = nb.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"混淆矩阵: \n{conf_matrix}")

    y_probs = nb.predict_proba(X_test)[:, 1]  # 选择概率中的正类概率
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
