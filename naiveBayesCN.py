import jieba

from itertools import islice

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tqdm.contrib.itertools import product
from naiveBayes import trainNB0, classifyNB


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


def trainNB1(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    if isinstance(trainMatrix, pd.DataFrame):
        numWords = len(trainMatrix.columns)
    else:
        numWords = len(trainMatrix[0])

    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = np.ones(numWords);
    p1Num = np.ones(numWords)
    p0Denom = 2.0;
    p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix.iloc[i] if isinstance(trainMatrix, pd.DataFrame) else trainMatrix[i]
            p1Denom += sum(trainMatrix.iloc[i]) if isinstance(trainMatrix, pd.DataFrame) else sum(trainMatrix[i])
        else:
            p0Num += trainMatrix.iloc[i] if isinstance(trainMatrix, pd.DataFrame) else trainMatrix[i]
            p0Denom += sum(trainMatrix.iloc[i]) if isinstance(trainMatrix, pd.DataFrame) else sum(trainMatrix[i])

    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)

    return p0Vect, p1Vect, pAbusive


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
