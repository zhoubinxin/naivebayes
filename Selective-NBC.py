# 实现 Selective Bayes Network Classifier 的示例代码，它是贝叶斯网络的一种变体，用于分类任务。
# 这里使用高斯朴素贝叶斯模型作为基础模型。

# 首先，确保已安装 `scikit-learn` 库。你可以使用以下命令进行安装：
# pip install scikit-learn

# 然后，使用下面的代码实现选择贝叶斯网络分类器：

import numpy as np
from joblib import Parallel, delayed
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from naiveBayesCN import load_stop_words, loadCNDataSet


# 定义 Selective Bayes Network Classifier 类
class SelectiveBayesNetworkClassifier:
    def __init__(self):
        self.base_classifier = None
        self.selected_features = None

    # 训练基础分类器
    def fit_base_classifier(self, X_train, y_train):
        self.base_classifier = GaussianNB()
        self.base_classifier.fit(X_train[:, self.selected_features], y_train)

    # 选择特征
    def select_features(self, X_train, y_train):
        total_features = X_train.shape[1]

        def evaluate_feature(i):
            selected_features = list(range(total_features))
            selected_features.remove(i)
            classifier = GaussianNB()
            classifier.fit(X_train[:, selected_features], y_train)
            y_pred = classifier.predict(X_train[:, selected_features])
            accuracy = accuracy_score(y_train, y_pred)
            return accuracy

        accuracies = Parallel(n_jobs=-1)(delayed(evaluate_feature)(i) for i in tqdm(range(total_features), desc="Selecting Features"))
        self.selected_features = [i for i in range(total_features) if accuracies[i] >= max(accuracies)]

    # 训练分类器
    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        self.select_features(X_train, y_train)
        self.fit_base_classifier(X_train, y_train)

    # 预测
    def predict(self, X_test):
        X_test = np.array(X_test)
        return self.base_classifier.predict(X_test[:, self.selected_features])

class SimpleCountVectorizer:
    def __init__(self, max_df=1.0, min_df=0.1):
        self.vocabulary_ = {}
        self.feature_names_ = []
        self.max_df = max_df
        self.min_df = min_df

    def fit(self, raw_documents):
        vocab = {}
        doc_count = {}
        total_docs = len(raw_documents)

        for doc in tqdm(raw_documents, desc="Fitting Vectorizer"):
            words = set(doc)  # 使用 jieba 分词
            unique_words = set(words)
            for word in unique_words:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    doc_count[word] = 1
                else:
                    doc_count[word] += 1

        self.vocabulary_ = {word: idx for word, idx in vocab.items()
                            if self.min_df <= doc_count[word] <= self.max_df * total_docs}
        self.feature_names_ = list(self.vocabulary_.keys())
        return self
    def transform(self, raw_documents):
        rows = []
        for doc in tqdm(raw_documents, desc="Transforming Documents"):
            words = set(doc)  # 使用 jieba 分词
            row = [0] * len(self.vocabulary_)
            for word in words:
                if word in self.vocabulary_:
                    row[self.vocabulary_[word]] += 1
            rows.append(row)
        return np.array(rows)

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def get_params(self, deep=True):
        return {"max_df": self.max_df, "min_df": self.min_df}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self



filename = "scu_stopwords"
stop_words = load_stop_words(filename)

lines = 5000  # 数据量
sample_size = 5000  # 类别样本数量
listOposts, listClasses = loadCNDataSet(lines, stop_words, sample_size)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(listOposts, listClasses, test_size=0.2, random_state=1)

# 向量化
vectorizer = SimpleCountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 创建并训练 Selective Bayes Network Classifier
classifier = SelectiveBayesNetworkClassifier()
classifier.fit(X_train_vec, y_train)

# 进行预测
y_pred = classifier.predict(X_test_vec)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 在这个示例中，首先定义了一个 SelectiveBayesNetworkClassifier 类。
# 在 fit 方法中，使用 select_features 方法选择具有最高准确率的特征子集，
# 然后使用 fit_base_classifier 方法训练基础分类器（这里使用高斯朴素贝叶斯）。
# 最后，使用 predict 方法进行预测。
