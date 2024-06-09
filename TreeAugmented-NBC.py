# Tree Augmented Naive Bayesian Classifier (TAN) 是一种改进的朴素贝叶斯分类器，
# 它在朴素贝叶斯的基础上引入了一个树结构来建模特征之间的依赖关系。下面是一个简单的示例，演示了如何使用Python实现TAN分类器。

import numpy as np
from collections import defaultdict

from mlxtend.evaluate import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from naiveBayesCN import *


class TANClassifier:
    def __init__(self):
        self.class_counts = defaultdict(int)  # 类别计数
        self.feature_counts = defaultdict(lambda: defaultdict(int))  # 特征计数
        self.feature_pair_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # 特征对计数
        self.class_feature_pair_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # 类别和特征对计数

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for i in range(n_samples):
            label = y[i]
            self.class_counts[label] += 1
            for j in range(n_features):
                feature = X[i][j]
                self.feature_counts[j][feature] += 1
                self.class_feature_pair_counts[label][j][feature] += 1
                if j > 0:  # 计算特征对计数
                    prev_feature = X[i][j-1]
                    self.feature_pair_counts[j-1][prev_feature][feature] += 1

    def predict(self, X):
        n_samples, n_features = X.shape
        predictions = []
        for i in range(n_samples):
            probs = {}
            for label in self.class_counts:
                class_prob = self.class_counts[label] / sum(self.class_counts.values())
                feature_probs = []
                for j in range(n_features):
                    if j == 0:
                        # 计算朴素贝叶斯部分的概率
                        feature_prob = self.class_feature_pair_counts[label][j][X[i][j]] / self.class_counts[label]
                    else:
                        # 计算TAN部分的概率
                        prev_feature = X[i][j-1]
                        feature = X[i][j]
                        pair_count = self.feature_pair_counts[j-1][prev_feature][feature]
                        feature_pair_count = self.feature_counts[j-1][prev_feature]
                        if feature_pair_count == 0:  # 处理分母为0的情况
                            feature_prob = 0
                        else:
                            feature_prob = pair_count / feature_pair_count
                    feature_probs.append(feature_prob)
                probs[label] = class_prob * np.prod(feature_probs)
            predictions.append(max(probs, key=probs.get))
        return predictions

filename = "scu_stopwords"
stop_words = load_stop_words(filename)

lines = 10000  # 数据量
sample_size = 10000  # 类别样本数量
listOposts, listClasses = loadCNDataSet(lines, stop_words, sample_size)
X_train, X_test, y_train, y_test = train_test_split(listOposts, listClasses, test_size=0.2, random_state=1)

vectorizer = SimpleTfidfVectorizer()  # 可以切换为 SimpleCountVectorizer
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 创建并训练TAN分类器
tan_classifier = TANClassifier()
tan_classifier.fit(X_train_vec, y_train)

# 进行预测
predictions = tan_classifier.predict(X_test_vec)

# 评估模型
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f"准确率: {accuracy}")
print(f"精确率: {precision}")
print(f"召回率: {recall}")
print(f"F1值: {f1}")

# 在这个示例中，我们定义了一个 `TANClassifier` 类来实现TAN分类器。
# 在 `fit` 方法中，我们计算了类别、特征以及特征对之间的计数。
# 在 `predict` 方法中，我们使用这些计数来计算预测的概率。