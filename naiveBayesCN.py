import jieba
from itertools import islice
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tqdm.contrib.itertools import product
from naiveBayes import trainNB0, classifyNB
import re
from collections import Counter
from sklearn.model_selection import KFold, train_test_split
from sklearn.base import clone
import numpy as np


# 简单朴素贝叶斯分类器
class SimpleNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)
            self.feature_log_prob_[idx] = np.log((X_c.sum(axis=0) + self.alpha) / (X_c.sum() + self.alpha * n_features))

    def predict(self, X):
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class SimpleCountVectorizer:
    def __init__(self):
        self.vocabulary_ = {}

    def fit(self, documents):
        vocabulary = set()
        for doc in documents:
            words = re.findall(r'\b\w+\b', doc.lower())
            vocabulary.update(words)
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(vocabulary))}

    def transform(self, documents):
        rows = []
        for doc in documents:
            word_count = Counter(re.findall(r'\b\w+\b', doc.lower()))
            row = np.zeros(len(self.vocabulary_))
            for word, count in word_count.items():
                if word in self.vocabulary_:
                    row[self.vocabulary_[word]] = count
            rows.append(row)
        return np.array(rows)

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)


class SimpleTfidfVectorizer(SimpleCountVectorizer):
    def transform(self, documents):
        count_matrix = super().transform(documents)
        row_sums = np.sum(count_matrix, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除以零
        tf = count_matrix / row_sums
        df = np.sum(count_matrix > 0, axis=0)
        idf = np.log((1 + len(documents)) / (1 + df)) + 1
        return tf * idf


class SimpleGridSearchCV:
    def __init__(self, estimator, param_grid, cv=5):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []

    def fit(self, X, y):
        self.best_score_ = -np.inf
        self.best_params_ = None

        param_grid_list = list(self.param_grid.values())[0]
        param_name = list(self.param_grid.keys())[0]

        for params in tqdm(param_grid_list, desc="超参数搜索"):
            scores = []
            for fold in range(self.cv):
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 / self.cv, random_state=fold)
                model = self.estimator.set_params(**{param_name: params})
                model.fit(X_train, y_train)
                scores.append(model.score(X_val, y_val))
            avg_score = np.mean(scores)
            if avg_score > self.best_score_:
                self.best_score_ = avg_score
                self.best_params_ = {param_name: params}

    def _param_iterator(self, param_grid):
        import itertools
        keys = param_grid.keys()
        values = (param_grid[key] for key in keys)
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))


def load_stop_words():
    """
    加载停用词列表
    """
    stop_words = set()
    with open('./data/cnsmss/stopWord.txt', 'r', encoding='utf-8') as file:
        for line in file:
            stop_words.add(line.strip())
    return stop_words


def loadDataSet(stop_words, lines=5000):
    """
    读取中文数据集并进行预处理
    """
    postingList = []  # 存储文本
    classVec = []  # 存储标签
    with open('./data/cnsmss/80w.txt', 'r', encoding='utf-8') as file:
        dataSet = [line.strip().split('\t') for line in islice(file, lines)]
        for item in tqdm(dataSet, desc='加载数据集：'):
            # 检查数据格式是否正确，至少包含3个元素
            if len(item) >= 3:
                classVec.append(int(item[1]))  # 假设第2个元素是类别
                # 去除停用词
                words = jieba.lcut(item[2], cut_all=False)
                postingList.append([word for word in words if word not in stop_words])
            else:
                print(f"警告：数据行格式不正确，已跳过。原始行: '{item}'")
    return postingList, classVec


def trainNB1(trainMatrix, trainCategory, alpha=1.0):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords) * alpha
    p1Num = np.ones(numWords) * alpha
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB0(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数
    :param vec2Classify: 待分类的文本向量
    :param p0Vec: 非垃圾词汇的概率
    :param p1Vec: 垃圾词汇的概率
    :param pClass1: 垃圾短信的概率
    :return: 分类结果
    """
    # 元素相乘
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1  # 垃圾信息
    else:
        return 0  # 正常信息


def grid_search_naive_bayes(X_train, y_train, X_test, y_test, param_grid):
    """
    手动实现的参数搜索函数
    :param X_train: 训练数据
    :param y_train: 训练标签
    :param X_test: 测试数据
    :param y_test: 测试标签
    :param param_grid: 参数网格
    :return: 最佳参数和对应的模型性能
    """
    best_score = 0
    best_params = None

    for params in tqdm(list(product(*param_grid.values())), desc='参数搜索'):
        max_df, alpha = params
        vectorizer = CountVectorizer(max_df=max_df)
        X_train_vec = vectorizer.fit_transform(X_train).toarray()
        X_test_vec = vectorizer.transform(X_test).toarray()

        p0V, p1V, pAb = trainNB0(X_train_vec, y_train)
        y_pred = [classifyNB(vec, p0V, p1V, pAb) for vec in X_test_vec]

        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_score:
            best_score = accuracy
            best_params = {'max_df': max_df, 'alpha': alpha}

    return best_params, best_score


def preprocess_doc(args):
    """
    单个文档预处理函数，用于多进程调用
    """
    doc, stop_words = args
    return ' '.join(jieba.lcut(doc, cut_all=False) if isinstance(doc, str) else doc)  # 预处理文档并返回处理后的文本字符串


if __name__ == '__main__':
    # 测试nativeBayesCN
    stop_words = load_stop_words()
    listOposts, listClasses = loadDataSet(stop_words)
    print("Data loaded.")
