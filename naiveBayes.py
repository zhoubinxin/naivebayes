import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import product

# 词表到向量的转换函数
def loadDataSet():
    """
    读取数据
    :return: postingList: 词条切分后的文档集合
             classVec: 类别标签
    """
    postingList = []  # 存储文本
    classVec = []  # 存储标签
    with open('./data/smss/SMSSpamCollection', 'r', encoding='utf-8') as file:
        dataSet = [line.strip().split('\t') for line in file.readlines()]

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    for item in tqdm(dataSet, desc='加载数据'):
        # ham -> 0：表示非垃圾短信
        # spam -> 1：表示垃圾短信
        if item[0] == 'ham':
            classVec.append(0)
        else:
            classVec.append(1)

        # 将每条短信拆分为单词列表
        words = re.split(r'\W+', item[1])
        # 移除空字符串并转换为小写，移除停用词
        words = [word.lower() for word in words if word != '' and word.lower() not in stop_words]
        postingList.append(words)

    return postingList, classVec

def createVocabList(dataSet):
    """
    提取数据集中的单词列表，去除重复的单词
    :param dataSet: 数据集
    :return: list(vocabSet)：返回一个包含所有文档中出现的不重复词的列表
    """
    # 创建一个空集
    vocabSet = set([])
    for document in tqdm(dataSet, desc='创建词表'):
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    """
    词表到向量的转换
    :param vocabList: 词表
    :param inputSet: 文档
    :return: returnVec: 文档向量
    """
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)

    # 遍历文档中的所有单词
    for word in inputSet:
        if word in vocabList:
            # 如果词表中的单词在输入文档中出现，则将returnVec中对应位置的值设为1
            returnVec[vocabList.index(word)] = 1
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 由文本向量组成的矩阵
    :param trainCategory: 训练样本对应的标签
    :return: p0Vec: 非垃圾词汇的概率
             p1Vec: 垃圾词汇的概率
             pAbusive: 垃圾短信的概率
    """
    numTrainDocs = len(trainMatrix)  # 文档个数
    numWords = len(trainMatrix[0])  # 单词个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算垃圾短信的概率
    # 初始化概率
    # 拉普拉斯平滑
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = numWords
    p1Denom = numWords

    # 遍历所有文档，统计每个单词在垃圾短信和非垃圾短信中出现的次数
    for i in range(numTrainDocs):
        if trainCategory[i]:
            # 向量相加
            p1Num += trainMatrix[i]  # 垃圾短信中出现该词的次数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]  # 非垃圾短信中出现该词的次数
            p0Denom += sum(trainMatrix[i])
    # 对每个元素做除法求概率，为了避免下溢出的影响，对计算结果取自然对数
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
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


def preprocess_doc(args):
    doc, stop_words = args
    words = re.split(r'\W+', doc)
    words = [word.lower() for word in words if word != '' and word.lower() not in stop_words]
    return ' '.join(words)

def load_stop_words():
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    return stop_words
