import re
import nltk
import numpy as np
from tqdm import tqdm
from nltk.corpus import stopwords


def loadDataSet():
    """
    读取数据
    :return: postingList: 词条切分后的文档集合
             classVec: 类别标签
    """
    docs = []  # 存储文本
    label = []  # 存储标签
    with open('./data/smss/SMSSpamCollection', 'r', encoding='utf-8') as file:
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

        # 将每条短信拆分为单词列表
        words = re.findall(r'\b\w+\b', item[1])
        # 转换为小写，移除停用词
        words = [word.lower() for word in words if word.lower() not in stop_words]
        docs.append(words)

    return docs, label


def loadTestDataSet():
    docs = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
            ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
            ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
            ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
            ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
            ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    label = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字，0 代表正常言论
    return docs, label


def createVocabList(docs):
    """
    提取数据集中的单词列表，去除重复的单词
    :param docs: 文档集合
    :return: list(vocabSet)：返回一个包含所有文档中出现的不重复词的列表
    """
    # 创建一个空集
    vocabSet = set()
    for doc in tqdm(docs, desc='创建词表'):
        # 创建两个集合的并集
        vocabSet.update(doc)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    词集模型
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


def bagOfWords2VecMN(vocabList, inputSet):
    """
    词袋模型
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


class TFIDF(object):
    def __init__(self, docs, vocabList):
        self.docs = docs
        self.vocabList = vocabList

    def calc_tf(self):
        """
        计算词频 TF
        :return: tfDicts: 包含每个文档的TF词典的列表
        """
        tfList = []
        for doc in tqdm(self.docs, desc='计算TF'):
            tfDoc = [0] * len(self.vocabList)
            word_count = len(doc)
            if word_count == 0:
                tfList.append(tfDoc)
                continue
            for word in doc:
                if word in self.vocabList:
                    index = self.vocabList.index(word)
                    tfDoc[index] += 1
            # 将每个词的频数除以文档中的总词数，得到词频
            tfDoc = [count / word_count for count in tfDoc]
            tfList.append(tfDoc)
        return tfList

    def calc_idf(self):
        """
        计算逆文档频率 IDF
        :return: idfDict: 包含每个单词的IDF值的词典
        """
        numDocs = len(self.docs)
        idfList = [0] * len(self.vocabList)

        for doc in tqdm(self.docs, desc='计算IDF'):
            unique_words = set(doc)
            for word in unique_words:
                if word in self.vocabList:
                    index = self.vocabList.index(word)
                    idfList[index] += 1

        # 计算每个单词的IDF值
        idfList = [np.log(numDocs / (1 + count)) for count in idfList]
        return idfList

    def calc_tfidf(self):
        """
        TF-IDF算法实现
        :return: returnVec: 文档向量
        """
        tfList = self.calc_tf()
        idfList = self.calc_idf()
        tfidfList = []

        for tfDoc in tqdm(tfList, desc='计算TF-IDF'):
            tfidfDoc = [tf * idf for tf, idf in zip(tfDoc, idfList)]
            tfidfDoc = self.mm(tfidfDoc)
            tfidfList.append(tfidfDoc)

        return tfidfList

    def mm(self, vec):
        """
        归一化向量 使得某一个特征对最终结果不会造成更大的影响
        :param vec: 向量
        :return: 归一化后的向量
        """
        minVal = min(vec)
        maxVal = max(vec)

        if maxVal == minVal:  # 避免除以0的情况
            return vec

        mx = 1
        mi = 0
        data = []
        for x in vec:
            X = (x - minVal) / (maxVal - minVal)
            data.append(X * (mx - mi) + mi)
        return data


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


def testingNB():
    """
    测试朴素贝叶斯分类器
    :return: 输出测试结果
    """
    docs, label = loadTestDataSet()
    vocabList = createVocabList(docs)
    tfidf = TFIDF(docs, vocabList)
    trainMat = []
    # for postinDoc in docs:
    #     trainMat.append(setOfWords2Vec(vocabList, postinDoc))
    trainMat = tfidf.calc_tfidf()
    print(trainMat)
    p0V, p1V, pAb = trainNB0(trainMat, label)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(vocabList, testEntry)
    print(testEntry, 'calssified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


def main():
    testingNB()


if __name__ == '__main__':
    main()
