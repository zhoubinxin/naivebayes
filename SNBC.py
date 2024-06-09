# 半朴素贝叶斯分类器是朴素贝叶斯分类器的一种扩展，它允许一些属性之间的依赖关系，而不是像朴素贝叶斯那样假设所有属性之间相互独立。
# 下面是一个简单的Python程序，实现了半朴素贝叶斯分类器：

import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split

from naiveBayesCN import load_stop_words, loadCNDataSet, SimpleCountVectorizer, SimpleGridSearchCV



class SemiNaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        X = np.array(X)  # 转换为 NumPy 数组
        y = np.array(y)  # 转换为 NumPy 数组
        self.classes = np.unique(y)
        self.class_probs = dict()
        self.feature_probs = dict()

        # 计算类别的先验概率
        total_samples = len(y)
        for c in self.classes:
            class_samples = X[y == c]
            self.class_probs[c] = (len(class_samples) + self.alpha) / (total_samples + len(self.classes) * self.alpha)

        # 计算每个特征在给定类别下的条件概率
        for c in self.classes:
            class_samples = X[y == c]
            self.feature_probs[c] = dict()
            for i in range(X.shape[1]):
                feature_values = class_samples[:, i]
                value_counts = Counter(feature_values)
                total_count = len(feature_values)
                self.feature_probs[c][i] = {value: (count + self.alpha) / (total_count + len(value_counts) * self.alpha) for value, count in value_counts.items()}

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def predict(self, X):
        X = np.array(X)  # 确保 X 是 NumPy 数组
        predictions = []
        for sample in X:
            max_class = None
            max_prob = -1
            for c in self.classes:
                class_prob = self.class_probs[c]
                conditional_prob = 1
                for i, value in enumerate(sample):
                    if i in self.feature_probs[c]:
                        if value in self.feature_probs[c][i]:
                            conditional_prob *= self.feature_probs[c][i][value]
                        else:
                            conditional_prob *= self.alpha / (len(self.feature_probs[c][i]) * self.alpha + len(X))
                    else:
                        conditional_prob *= self.alpha / (len(X) + self.alpha)
                posterior_prob = class_prob * conditional_prob
                if posterior_prob > max_prob:
                    max_prob = posterior_prob
                    max_class = c
            predictions.append(max_class)
        return predictions


# 示例用法
if __name__ == "__main__":
    # 示例数据集
    filename = "scu_stopwords"
    stop_words = load_stop_words(filename)

    lines = 10000  # 数据量
    sample_size = 10000  # 类别样本数量
    listOposts, listClasses = loadCNDataSet(lines, stop_words, sample_size)


    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(listOposts, listClasses, test_size=0.2, random_state=1)

    vectorizer = SimpleCountVectorizer()  # 可以切换为 SimpleCountVectorizer
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)


    # 使用 SimpleGridSearchCV 进行超参数搜索
    param_grid = {'alpha': [0.3, 0.5]}
    grid_search = SimpleGridSearchCV(SemiNaiveBayesClassifier(), param_grid, cv=2)
    grid_search.fit(X_train_vec, y_train)

    # 得到最佳参数
    best_params = grid_search.best_params_
    print(f"最佳参数: {best_params}")

    # 使用最佳参数训练模型
    # best_model = SimpleNaiveBayes(4.0)
    best_model = SemiNaiveBayesClassifier(**best_params)
    best_model.fit(X_train_vec, y_train)
    # 训练模型
    model = SemiNaiveBayesClassifier()
    model.fit(X_train, y_train)

    # 进行预测
    predictions = model.predict(X_test)
    print("Predictions:", predictions)

# 在这个示例中，首先定义了一个`SemiNaiveBayesClassifier`类，它具有`fit`和`predict`方法。`fit`方法用于训练模型，`predict`方法用于进行预测。

# 在`fit`方法中，首先计算类别的先验概率，然后计算每个特征在给定类别下的条件概率。在`predict`方法中，使用贝叶斯定理来计算每个类别的后验概率，并选择具有最大后验概率的类别作为预测结果。

# 实际应用中可能需要更复杂的数据处理和模型调优。

