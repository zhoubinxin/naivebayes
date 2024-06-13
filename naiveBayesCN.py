import nltk
from nltk.corpus import stopwords
from tqdm.contrib.itertools import product
# from CNCountVectorizer import tokenizer
from joblib import Parallel, delayed
import itertools
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import islice
from tqdm import tqdm
import jieba
import re
import nltk
from nltk.corpus import stopwords
from joblib import Parallel, delayed
import itertools

from sklearn.model_selection import train_test_split

from naiveBayes import downsample


class SimpleSPODE(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 拉普拉斯平滑参数
        self.feature_probs = {}
        self.class_probs = {}
        self.parent = None

    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        self.class_probs = {c: count / len(y) for c, count in zip(self.classes, class_counts)}

        # 选择特定特征作为父特征（简单起见，选择第一个特征）
        self.parent = 0
        self.feature_probs = {c: {} for c in self.classes}

        for c in self.classes:
            X_c = X[y == c]
            parent_vals, parent_counts = np.unique(X_c[:, self.parent], return_counts=True)
            parent_probs = {val: (count + self.alpha) / (len(X_c) + self.alpha * len(parent_vals))
                            for val, count in zip(parent_vals, parent_counts)}
            self.feature_probs[c][self.parent] = parent_probs

            for feature in range(X.shape[1]):
                if feature == self.parent:
                    continue
                self.feature_probs[c][feature] = {}
                for parent_val in parent_vals:
                    X_parent = X_c[X_c[:, self.parent] == parent_val]
                    feature_vals, feature_counts = np.unique(X_parent[:, feature], return_counts=True)
                    self.feature_probs[c][feature][parent_val] = {
                        val: (count + self.alpha) / (len(X_parent) + self.alpha * len(feature_vals))
                        for val, count in zip(feature_vals, feature_counts)}

    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {c: np.log(self.class_probs[c]) for c in self.classes}
            parent_val = x[self.parent]
            for c in self.classes:
                parent_prob = self.feature_probs[c][self.parent].get(parent_val, self.alpha / (
                            sum(self.feature_probs[c][self.parent].values()) + self.alpha))
                for feature in range(X.shape[1]):
                    if feature == self.parent:
                        continue
                    feature_val = x[feature]
                    feature_prob = self.feature_probs[c][feature].get(parent_val, {}).get(feature_val, self.alpha / (
                                sum(self.feature_probs[c][feature].get(parent_val, {}).values()) + self.alpha))
                    class_probs[c] += np.log(parent_prob * feature_prob)
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


class BertNaiveBayes(object):
    def __init__(self, alpha=1.0):
        """
        初始化朴素贝叶斯分类器
        """
        self.alpha = alpha
        self.classes = None
        self.classLogPrior = None
        self.featureLogProb = None
        self.pAbusive = None

    def fit(self, trainMat, label):
        """
        朴素贝叶斯分类器训练函数
        :param trainMat: 由文本向量组成的矩阵
        :param label: 训练样本对应的标签
        :return: p0Vec: 非垃圾词汇的概率
                 p1Vec: 垃圾词汇的概率
                 pAbusive: 垃圾短信的概率
        """
        trainMat = np.array(trainMat)
        numTrainDocs, numWords = trainMat.shape  # 文档个数,单词个数
        self.classes = np.unique(label)  # 类别标签
        numClasses = len(self.classes)  # 类别个数

        self.pAbusive = sum(label) / float(numTrainDocs)  # 计算垃圾短信的概率

        self.classLogPrior = np.zeros(numClasses)  # 初始化先验概率
        self.featureLogProb = np.zeros((numClasses, numWords))  # 初始化条件概率

        for idx, c in enumerate(self.classes):
            X_c = trainMat[label == c]  # 获取类别为c的样本
            self.classLogPrior[idx] = np.log(len(X_c) / float(numTrainDocs))  # 计算先验概率
            # 计算条件概率时添加平滑处理
            smoothed_prob = (X_c.sum(axis=0) + self.alpha) / (X_c.sum() + self.alpha * numWords)
            smoothed_prob[smoothed_prob == 0] = 1e-9  # 将概率为零的部分替换为一个很小的数值，例如 1e-9
            self.featureLogProb[idx] = np.log(smoothed_prob)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def predict(self, vec2Classify):
        """
        朴素贝叶斯分类函数
        :param vec2Classify: 待分类的文本向量
        :return: 分类结果
        """
        jll = vec2Classify @ self.featureLogProb.T + self.classLogPrior
        return self.classes[np.argmax(jll, axis=1)]

    def predict_proba(self, vec2Classify):
        """
        返回每个样本属于每个类的概率
        :param vec2Classify: 待分类的文本向量
        :return: 每个样本属于每个类的概率
        """
        jll = vec2Classify @ self.featureLogProb.T + self.classLogPrior
        log_prob_x = np.log(np.sum(np.exp(jll), axis=1))
        return np.exp(jll - log_prob_x[:, np.newaxis])

    def setAlpha(self, alpha):
        self.alpha = alpha
        return self


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

    def predict_proba(self, X):
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        log_prob_x = np.log(np.sum(np.exp(jll), axis=1))
        return np.exp(jll - log_prob_x[:, np.newaxis])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# def plot_ks_curve(y_true, y_pred_prob):
#
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
#     ks_statistic = max(tpr - fpr)
#     ks_threshold = thresholds[np.argmax(tpr - fpr)]
#
#     plt.figure()
#     plt.plot(thresholds, tpr, label='真阳性率')
#     plt.plot(thresholds, fpr, label='假阳性率')
#     plt.axvline(ks_threshold, color='red', linestyle='--', label=f'KS阈值: {ks_threshold:.2f}')
#     plt.axhline(ks_statistic, color='blue', linestyle='--', label=f'KS统计量: {ks_statistic:.2f}')
#     plt.xlabel('Threshold')
#     plt.ylabel('Rate')
#     plt.title('KS Curve')
#     plt.legend(loc='best')
#
#     # 设置窗口标题
#     fig = plt.gcf()
#     fig.canvas.manager.set_window_title('KS 曲线')
#
#     plt.show()

import numpy as np

class SimpleBNaiveBayes:
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
            class_count = X_c.shape[0]
            feature_count = X_c.sum(axis=0)
            total_feature_count = feature_count.sum()

            # 确保分母不为零
            denominator = total_feature_count + self.alpha * n_features
            if denominator == 0:
                denominator = 1  # 避免分母为零

            self.class_log_prior_[idx] = np.log(class_count / n_samples)
            self.feature_log_prob_[idx] = np.log((feature_count + self.alpha) / (denominator + np.finfo(float).eps))

    def predict(self, X):
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_proba(self, X):
        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        log_prob_x = np.log(np.sum(np.exp(jll), axis=1))
        return np.exp(jll - log_prob_x[:, np.newaxis])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_params(self, deep=True):
        return {"alpha": self.alpha}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self



class SimpleCountVectorizer:
    def __init__(self, max_df=1.0, min_df=1):
        self.vocabulary_ = {}
        self.feature_names_ = []
        self.max_df = max_df
        self.min_df = min_df

    def fit(self, raw_documents):
        vocab = {}
        doc_count = {}
        total_docs = len(raw_documents)

        for doc in raw_documents:
            words = set(doc)  # 使用 jieba 分词
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    doc_count[word] = 1
                else:
                    doc_count[word] += 1

        # 过滤词汇
        self.vocabulary_ = {word: idx for word, idx in vocab.items()
                            if self.min_df <= doc_count[word] <= self.max_df * total_docs}
        self.feature_names_ = list(self.vocabulary_.keys())
        return self

    def transform(self, raw_documents):
        rows = []
        for doc in raw_documents:
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


class SimpleTfidfVectorizer:
    def __init__(self):
        self.vocabulary_ = {}
        self.idf_ = {}

    def fit(self, raw_documents):
        vocab = {}
        doc_count = {}
        total_docs = len(raw_documents)

        for doc in raw_documents:
            words = doc  # 使用 jieba 分词
            for word in words:
                if word not in vocab:
                    vocab[word] = len(vocab)
                    doc_count[word] = 1
                else:
                    doc_count[word] += 1

        self.vocabulary_ = vocab

        for word, count in doc_count.items():
            self.idf_[word] = np.log(total_docs / (1 + count))

        return self

    def transform(self, raw_documents):
        rows = []
        for doc in raw_documents:
            words = doc  # 使用 jieba 分词
            row = [0] * len(self.vocabulary_)
            word_count = {}
            for word in words:
                if word in self.vocabulary_:
                    word_count[word] = word_count.get(word, 0) + 1

            for word, count in word_count.items():
                if word in self.idf_:
                    row[self.vocabulary_[word]] = count * self.idf_[word]

            rows.append(row)
        return np.array(rows)

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)


class SimpleGridSearchCV:
    def __init__(self, estimator, param_grid, cv, n_jobs=1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []

    def fit(self, X, y):
        self.best_score_ = -np.inf
        self.best_params_ = None

        param_combinations = list(self._param_iterator(self.param_grid))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_params)(params, X, y) for params in tqdm(param_combinations, desc="Grid Search")
        )

        for params, avg_score in results:
            self.cv_results_.append({'params': params, 'score': avg_score})
            if avg_score > self.best_score_:
                self.best_score_ = avg_score
                self.best_params_ = params

        # 保存结果到文件
        self._save_results_to_file()

    def _evaluate_params(self, params, X, y):
        scores = []
        for fold in range(self.cv):

            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 / self.cv, random_state=fold)
            model = self.estimator.set_params(**params)
            model.fit(X_train, y_train)
            scores.append(model.score(X_val, y_val))
        avg_score = np.mean(scores)
        return params, avg_score

    def _param_iterator(self, param_grid):
        keys = param_grid.keys()
        values = (param_grid[key] for key in keys)
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def _save_results_to_file(self):
        os.makedirs('result', exist_ok=True)
        with open('result/cv_results.txt', 'w', encoding='utf-8') as file:
            for result in self.cv_results_:
                file.write(f"Params: {result['params']}, Score: {result['score']}\n")


def load_stop_words(filename):
    """
    加载停用词列表
    """
    stop_words = set()
    with open(f'./data/cnsmss/{filename}.txt', 'r', encoding='utf-8') as file:
        for line in file:
            stop_words.add(line.strip())
    return stop_words


class SimpleHalvingGridSearchCV:
    def __init__(self, estimator, param_grid, cv=5, factor=3, min_resources='exhaust', n_jobs=-1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.factor = factor
        self.min_resources = min_resources
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []

    def fit(self, X, y):
        self.best_score_ = -np.inf
        self.best_params_ = None

        param_combinations = list(self._param_iterator(self.param_grid))
        n_candidates = len(param_combinations)
        print(f"参数组合数量: {n_candidates}")

        # 初始资源数
        n_resources = len(X) if self.min_resources == 'exhaust' else self.min_resources
        if isinstance(n_resources, str) and n_resources == 'exhaust':
            n_resources = max(len(X) // self.factor, 1)
        print(f"初始资源数量: {n_resources}")

        while n_candidates > 1 and n_resources <= len(X):
            print(f"评估 {n_candidates} 个候选参数，每个使用 {n_resources} 个资源。")
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._evaluate_params)(params, X, y, n_resources) for params in param_combinations
            )

            # 按得分排序，选择得分最好的前1/factor个参数组合
            results.sort(key=lambda x: x[1], reverse=True)
            n_candidates = max(1, n_candidates // self.factor)
            param_combinations = [params for params, score in results[:n_candidates]]

            for result in results:
                self.cv_results_.append({'params': result[0], 'score': result[1]})

            n_resources = min(len(X), n_resources * self.factor)

            if results[0][1] > self.best_score_:
                self.best_score_ = results[0][1]
                self.best_params_ = results[0][0]

        print(f"最佳得分: {self.best_score_}")
        print(f"最佳参数: {self.best_params_}")

        # 保存结果到文件
        self._save_results_to_file()

    def _evaluate_params(self, params, X, y, n_resources):
        scores = []
        for fold in range(self.cv):
            X_train, X_val, y_train, y_val = train_test_split(X[:n_resources], y[:n_resources], test_size=1 / self.cv,
                                                              random_state=fold)
            model = self.estimator.set_params(**params)
            model.fit(X_train, y_train)
            scores.append(model.score(X_val, y_val))
        avg_score = np.mean(scores)
        return params, avg_score

    def _param_iterator(self, param_grid):
        keys = param_grid.keys()
        values = (param_grid[key] for key in keys)
        for combination in product(*values):
            yield dict(zip(keys, combination))

    def _save_results_to_file(self):
        os.makedirs('result', exist_ok=True)
        with open('result/cv_results.txt', 'w') as f:
            for result in self.cv_results_:
                f.write(f"Params: {result['params']}, Score: {result['score']}\n")


class SimpleGridSearchCVBert:
    def __init__(self, estimator, param_grid, cv, n_jobs=-1):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []

    def fit(self, X, y):
        self.best_score_ = -np.inf
        self.best_params_ = None

        param_combinations = list(self._param_iterator(self.param_grid))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_params)(params, X, y) for params in tqdm(param_combinations, desc="Grid Search")
        )

        for params, avg_score in results:
            self.cv_results_.append({'params': params, 'score': avg_score})
            if avg_score > self.best_score_:
                self.best_score_ = avg_score
                self.best_params_ = params

        self._save_results_to_file()

    def _evaluate_params(self, params, X, y):
        scores = []
        for fold in range(self.cv):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 / self.cv, random_state=fold)
            model = self.estimator.set_params(**params)
            model.fit(X_train, y_train)
            scores.append(model.score(X_val, y_val))
        avg_score = np.mean(scores)
        return params, avg_score

    def _param_iterator(self, param_grid):
        keys = param_grid.keys()
        values = (param_grid[key] for key in keys)
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def _save_results_to_file(self):
        os.makedirs('result', exist_ok=True)
        with open('result/cv_results.txt', 'w', encoding='utf-8') as file:
            for result in self.cv_results_:
                file.write(f"Params: {result['params']}, Score: {result['score']}\n")



import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import itertools
import os
from tqdm import tqdm

class SimpleBertGridSearchCV:
    def __init__(self, estimator, param_grid, cv, n_jobs=1):  # 默认 n_jobs 为 1
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []

    def fit(self, X, y):
        self.best_score_ = -np.inf
        self.best_params_ = None

        param_combinations = list(self._param_iterator(self.param_grid))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_params)(params, X, y) for params in tqdm(param_combinations, desc="Grid Search")
        )

        for params, avg_score in results:
            self.cv_results_.append({'params': params, 'score': avg_score})
            if avg_score > self.best_score_:
                self.best_score_ = avg_score
                self.best_params_ = params

        # 保存结果到文件
        self._save_results_to_file()

    def _evaluate_params(self, params, X, y):
        scores = []
        for fold in range(self.cv):
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1 / self.cv, random_state=fold)
            model = self.estimator.set_params(**params)
            model.fit(X_train, y_train)
            scores.append(model.score(X_val, y_val))
        avg_score = np.mean(scores)
        return params, avg_score

    def _param_iterator(self, param_grid):
        keys = param_grid.keys()
        values = (param_grid[key] for key in keys)
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))

    def _save_results_to_file(self):
        os.makedirs('result', exist_ok=True)
        with open('result/cv_results.txt', 'w', encoding='utf-8') as file:
            for result in self.cv_results_:
                file.write(f"Params: {result['params']}, Score: {result['score']}\n")




def loadCNDataSet(lines, stop_words):
    """
    读取中文数据集并下采样
    :return:
    """
    docs = []  # 存储文本
    labels = []  # 存储标签

    try:
        with open('./data/cnsmss/80w.txt', 'r', encoding='utf-8') as file:
            dataSet = [line.strip().split('\t') for line in islice(file, lines)]
    except:
        # GitHub action
        with open('naivebayes/data/cnsmss/80w.txt', 'r', encoding='utf-8') as file:
            dataSet = [line.strip().split('\t') for line in islice(file, lines)]

    for item in tqdm(dataSet, desc='加载数据集：'):
        # 0：非垃圾短信；1：垃圾短信
        labels.append(int(item[1]))

        # 将每条短信拆分为单词列表
        try:
            words = jieba.lcut(item[2], cut_all=False)
            # 去除空格
            words = [word for word in words if word != ' ' and word not in stop_words]
            docs.append(words)
        except IndexError as e:
            print('\n', e)
            pass

    # if sample_size:
    #     from collections import Counter
    #     import random
    #
    #     # 下采样
    #     counter = Counter(labels)
    #     min_class = min(counter, key=counter.get)
    #     min_count = counter[min_class]
    #
    #     if sample_size > min_count:
    #         sample_size = min_count
    #
    #     sampled_docs = []
    #     sampled_labels = []
    #
    #     for label in set(labels):
    #         label_docs = [doc for doc, l in zip(docs, labels) if l == label]
    #         sampled_docs.extend(random.sample(label_docs, sample_size))
    #         sampled_labels.extend([label] * sample_size)
    #
    #     docs = sampled_docs
    #     labels = sampled_labels


    return docs, labels

# 将文本数据转换为 BERT token IDs
def tokenize_text(text):
    return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=512)

def loadCNBertDataSet(lines, stop_words):
    docs = []  # 存储文本
    labels = []  # 存储标签

    try:
        with open('./data/cnsmss/80w.txt', 'r', encoding='utf-8') as file:
            dataSet = [line.strip().split('\t') for line in islice(file, lines)]
    except:
        # GitHub action
        with open('naivebayes/data/cnsmss/80w.txt', 'r', encoding='utf-8') as file:
            dataSet = [line.strip().split('\t') for line in islice(file, lines)]

    for item in tqdm(dataSet, desc='加载数据集：'):
        # 0：非垃圾短信；1：垃圾短信
        labels.append(int(item[1]))

        # 使用 BERT tokenizer 处理文本数据
        try:
            # 分词并去除停用词
            words = jieba.cut(item[2])
            filtered_words = [word for word in words if word not in stop_words]
            filtered_text = ' '.join(filtered_words)

            token_ids = tokenize_text(filtered_text)
            docs.append(token_ids)
        except IndexError as e:
            print('\n', e)
            pass

    return docs, labels

def loadBertDataSet(target_spam_count=None):
    """
    读取数据
    :return: postingList: 词条切分后的文档集合
             classVec: 类别标签
    """
    docs = []  # 存储文本
    label = []  # 存储标签
    try:
        with open('./data/smss/SMSSpamCollection', 'r', encoding='utf-8') as file:
            dataSet = [line.strip().split('\t') for line in file.readlines()]
    except:
        # GitHub action
        with open('naivebayes/data/smss/SMSSpamCollection', 'r', encoding='utf-8') as file:
            dataSet = [line.strip().split('\t') for line in file.readlines()]

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    for item in tqdm(dataSet, desc='加载数据'):
        # ham -> 0：表示非垃圾短信
        # spam -> 1：表示垃圾短信
        if item[0] == 'ham':
            label.append(0)
        else:
            label.append(1)

        # 数据预处理
        text = re.sub('', "'", item[1])
        text = text.lower()

        # 将每条短信拆分为单词列表
        words = re.findall(r'\b\w+\b', text)
        # 移除停用词
        words = [word for word in words if word not in stop_words]
        docs.append(words)

    if target_spam_count:
        # 下采样
        docs, label = downsample(docs, label, target_spam_count)

    # 确保文本和标签具有相同数量的样本
    assert len(docs) == len(label), "文本和标签样本数量不一致"

    return docs, label

if __name__ == '__main__':
    # 测试nativeBayesCN
    stop_words = load_stop_words()
    listOposts, listClasses = loadCNDataSet(5000)
    print("Data loaded.")
