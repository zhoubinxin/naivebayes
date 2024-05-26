from mlxtend.evaluate import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import islice
import nativeBayesCN as nbCN
import numpy as np


def load_stop_words():
    """加载停用词"""
    stop_words = set()
    with open('./data/cnsmss/stopWord.txt', 'r', encoding='utf-8') as file:
        for line in file:
            stop_words.add(line.strip())
    return stop_words


def load_dataset(stop_words, lines=5000):
    """
    读取并预处理中文数据集，包括去除停用词
    """
    postingList, classVec = [], []
    with open('./data/cnsmss/80w.txt', 'r', encoding='utf-8') as file:
        dataset = [line.strip().split('\t') for line in islice(file, lines)]
    for item in tqdm(dataset, desc='加载数据集：'):
        classVec.append(int(item[1]))
        words = nbCN.jieba_lcut(item[2], cut_all=False)
        words = [word for word in words if word not in stop_words]  # 去除停用词
        postingList.append(words)
    return postingList, classVec


def create_vocab_list(dataSet):
    """提取词汇表，去除重复项"""
    vocabSet = set()
    for document in dataSet:
        vocabSet.update(document)
    return list(vocabSet)


def set_of_words_to_vec(vocabList, inputSet):
    """将文档转换为词向量"""
    returnVec = [1 if word in inputSet else 0 for word in vocabList]
    return returnVec


def main():
    stop_words = load_stop_words()
    listOposts, listClasses = load_dataset(stop_words)
    myVocabList = create_vocab_list(listOposts)

    # 逐个文档构建词向量，避免一次性加载所有文档到内存
    trainMat = []
    for postinDoc in tqdm(listOposts, desc='构建词向量矩阵'):
        trainMat.append(set_of_words_to_vec(myVocabList, postinDoc))

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(trainMat, listClasses, test_size=0.2, random_state=1)

    # 训练朴素贝叶斯分类器
    p0V, p1V, pAb = nbCN.train_NB0(np.array(X_train), np.array(y_train))

    # 预测和评估
    y_pred = [nbCN.classify_NB(doc, p0V, p1V, pAb) for doc in X_test]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 打印结果
    print(f"准确率: {accuracy}")
    print(f"精确率: {precision}")
    print(f"召回率: {recall}")
    print(f"F1值: {f1}")

    # 保存结果到txt文件
    with open('result/score.txt', 'w', encoding='utf-8') as file:
        file.write(f"准确率: {accuracy}\n")
        file.write(f"精确率: {precision}\n")
        file.write(f"召回率: {recall}\n")
        file.write(f"F1值: {f1}\n")

if __name__ == '__main__':
    main()






