# naiveBayes算法
import re

from numpy import *
from tqdm import tqdm
import jieba


# 词表到向量的转换函数
def loadDataSet():
    """
    读取数据

    :return:
    """
    postingList = []  # 存储文本
    classVec = []  # 存储标签
    with open('./data/smss/SMSSpamCollection', 'r', encoding='utf-8') as file:
        dataSet = [line.strip().split('\t') for line in file.readlines()]

    for item in dataSet:
        # ham -> 0：表示非垃圾短信
        # spam -> 1：表示垃圾短信
        if item[0] == 'ham':
            classVec.append(1)
        else:
            classVec.append(0)

        # 将每条短信拆分为单词列表
        words = re.split(r'\W+', item[1])
        # 移除空字符串并转换为小写
        words = [word.lower() for word in words if word != '']  # 转换为小写并去除空字符串
        postingList.append(words)

    return postingList, classVec


def loadCNDataSet():
    """
    读取中文数据集

    :return:
    """
    postingList = []  # 存储文本
    classVec = []  # 存储标签
    with open('./data/cnsmss/80w.txt', 'r', encoding='utf-8') as file:
        dataSet = [line.strip().split('\t') for line in file.readlines()]

    for item in tqdm(dataSet, desc='Loading data...'):
        # 0：非垃圾短信；1：垃圾短信
        classVec.append(item[1])

        # 将每条短信拆分为单词列表
        try:
            words = jieba.lcut(item[2], cut_all=False)
            postingList.append(words)
        except IndexError as e:
            print('\n', e)
            pass

    return postingList, classVec


def loadTestDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字，0 代表正常言论
    return postingList, classVec


def createVocabList(dataSet):
    """
    提取数据集中的单词列表

    :param dataSet:
    :return:
    """
    # 创建一个空集
    vocabSet = set([])
    for document in tqdm(dataSet, desc='Creating vocabulary list...'):
        # 创建两个集合的并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    将输入的文本转换为向量形式，将输入集合中的词语在词表中对应的位置设置为 1
    :param vocabList: 词表
    :param inputSet: 输入集合
    :return:
    """
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(f"the word: {word} is not in my Vocabulary!")
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 由文本向量组成的矩阵
    :param trainCategory: 训练样本对应的标签
    :return:
    """
    numTrainDocs = len(trainMatrix)  # 训练样本的数量
    numWords = len(trainMatrix[0])  # 词表的大小
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算垃圾邮件的概率
    # 初始化概率
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 向量相加
            p1Num += trainMatrix[i]  # 垃圾邮件中出现该词的次数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]  # 非垃圾邮件中出现该词的次数
            p0Denom += sum(trainMatrix[i])
    # 对每个元素素做除法
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    """
    朴素贝叶斯分类函数
    :param vec2Classify: 待分类的文本向量
    :param p0Vec: 非垃圾邮件的概率
    :param p1Vec: 垃圾邮件的概率
    :param pClass1: 垃圾邮件的概率
    :return: 分类结果
    """
    # 元素相乘
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1  # 垃圾邮件
    else:
        return 0  # 正常邮件


def testingNB():
    """
    测试朴素贝叶斯分类器
    :return: 输出测试结果
    """
    listOPosts, listClasses = loadTestDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(testEntry, 'calssified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(testEntry, 'calssified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def bagOfWords2VecMN(vocabList, inputSet):
    """
    朴素贝叶斯词袋模型
    :param vocabList:
    :param inputSet:
    :return:
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def main():
    # listOposts, listClasses = loadDataSet()
    # listOposts, listClasses = loadTestDataSet()
    # myVocabList = createVocabList(listOposts)
    # trainMat = []
    # for postinDoc in tqdm(listOposts):
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #
    # print(trainMat)
    # p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print(pAb)
    # print(p0V)
    # print(p1V)
    # testingNB()
    loadCNDataSet()


if __name__ == '__main__':
    main()
