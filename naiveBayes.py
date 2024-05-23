# naiveBayes算法
import re
import string

import numpy as np
import pandas as pd
from nltk.corpus import wordnet
from tqdm import tqdm
import nltk
from nltk.stem import WordNetLemmatizer


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

    # 读取停用词
    stopwords = set()
    with open('./data/smss/stopword', 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file.readlines()])

    for item in tqdm(dataSet, desc='加载数据'):
        # ham -> 0：表示非垃圾短信
        # spam -> 1：表示垃圾短信
        if item[0] == 'ham':
            classVec.append(1)
        else:
            classVec.append(0)

        # 将每条短信拆分为单词列表
        words = re.split(r'\W+', item[1])
        # 移除空字符串并转换为小写，移除停用词
        words = [word.lower() for word in words if word != '' and word not in stopwords]
        postingList.append(words)

    return postingList, classVec


def loadDataSet2():
    """
    读取数据
    :return: postingList: 词条切分后的文档集合
             classVec: 类别标签
    """
    postingList = []  # 存储文本
    classVec = []  # 存储标签
    with open('./data/smss/SMSSpamCollection', 'r', encoding='utf-8') as file:
        dataSet = [line.strip().split('\t') for line in file.readlines()]

    for item in tqdm(dataSet, desc='加载数据'):
        # ham -> 0：表示非垃圾短信
        # spam -> 1：表示垃圾短信
        if item[0] == 'ham':
            classVec.append(1)
        else:
            classVec.append(0)

        tokens = nltk.word_tokenize(item[1])  # 分词
        # 去除标点
        tokens = [remove_punctuation(token) for token in tokens]
        # 过滤空字符串
        tokens = [token for token in tokens if token]
        # 转为小写
        tokens = [token.lower() for token in tokens]

        tagged_sent = nltk.pos_tag(tokens)  # 词性标注

        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原

        postingList.append(lemmas_sent)
    return postingList, classVec


def remove_punctuation(text):
    return ''.join([char for char in text if char not in string.punctuation])


def get_wordnet_pos(tag):
    """
    获取单词的词性
    :param tag:
    :return:
    """
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


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
        else:
            print(f"the word: {word} is not in my Vocabulary!")
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    """
    词袋到向量的转换
    :param vocabList: 词袋
    :param inputSet: 文档
    :return: returnVec: 文档向量
    """
    # 和词集模型相比，词袋模型会考虑词条出现的次数
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        # 如果文档中的单词在词汇表中，则相应向量位置加1
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def bagOfWords2VecTFIDF(vocabList, inputSet, postingList):
    """
    使用TF-IDF进行特征的权重修正的词袋模型
    :param postingList:
    :param vocabList:
    :param inputSet:
    :return:
    """

    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            wordIndex = vocabList.index(word)
            tf = calcTF(word, inputSet)
            idf = calcIDF(word, postingList)
            returnVec[wordIndex] = tf * idf
    return returnVec


def calcTF(word, inputSet):
    """
    计算词频 TF
    """
    return inputSet.count(word) / len(inputSet)


def calcIDF(word, docList):
    """
    计算逆文档频率 IDF
    """
    numDocsContainingWord = sum([1 for doc in docList if word in doc])
    return np.log(len(docList) / (1 + numDocsContainingWord))


def trainNB0(trainMatrix, trainCategory):
    """
    朴素贝叶斯分类器训练函数
    :param trainMatrix: 由文本向量组成的矩阵
    :param trainCategory: 训练样本对应的标签
    :return: p0Vec: 非垃圾词汇的概率
             p1Vec: 垃圾词汇的概率
             pAbusive: 垃圾邮件的概率
    """
    numTrainDocs = len(trainMatrix)  # 文档个数
    numWords = len(trainMatrix[0])  # 单词个数
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 计算垃圾邮件的概率
    # 初始化概率
    # 拉普拉斯平滑
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = numWords
    p1Denom = numWords

    # 遍历所有文档，统计每个单词在垃圾邮件和非垃圾邮件中出现的次数
    for i in range(numTrainDocs):
        if trainCategory[i]:
            # 向量相加
            p1Num += trainMatrix[i]  # 垃圾邮件中出现该词的次数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]  # 非垃圾邮件中出现该词的次数
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
    :param pClass1: 垃圾邮件的概率
    :return: 分类结果
    """
    # 元素相乘
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1  # 垃圾邮件
    else:
        return 0  # 正常邮件


def testingNB():
    """
    测试朴素贝叶斯分类器
    :return: 输出测试结果
    """
    listOPosts, listClasses = loadDataSet()
    # 保存listOPosts
    print(listOPosts)
    df = pd.DataFrame(listOPosts)
    df.to_csv('./data/listOPosts.csv', index=False)
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
    testingNB()


if __name__ == '__main__':
    main()
