import random
import re
from collections import Counter, defaultdict

import nltk
import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from nltk.corpus import stopwords


class NaiveBayes(object):
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
            self.featureLogProb[idx] = np.log(
                (X_c.sum(axis=0) + self.alpha) / (X_c.sum() + self.alpha * numWords))  # 计算条件概率

    def evaluateModel(self, X_test_vec, y_test):
        """
        评估模型
        :param X_test_vec: 测试样本
        :param y_test: 测试样本对应的标签
        :return:
        """
        y_pred = self.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)
        return accuracy, precision, recall, f1, conf_matrix

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


class WordToVec(object):
    def __init__(self):
        self.vocabList = []
        self.idf = []

    def fit(self, docs):
        """
        创建词汇表
        :param docs:
        :return:
        """
        self.vocabList = list(set([word for doc in docs for word in doc]))

    def fit_tfidf(self, docs):
        vocab_dict = {}
        doc_count = {}
        total_docs = len(docs)

        for doc in docs:
            words = set(doc)
            for word in words:
                if word not in vocab_dict:
                    vocab_dict[word] = len(vocab_dict)
                    doc_count[word] = 1
                else:
                    doc_count[word] += 1

        self.vocabList = [None] * len(vocab_dict)
        for word, index in vocab_dict.items():
            self.vocabList[index] = word

        self.idf = [0] * len(vocab_dict)
        for word, count in doc_count.items():
            index = vocab_dict[word]
            self.idf[index] = np.log(total_docs / (1 + count))

        return self

    def calc_idf(self, docs):
        """
        计算逆文档频率 IDF
        :param docs: List[List[str]] - 文档集合
        :return: List[float] - 包含每个单词的IDF值的列表
        """
        numDocs = len(docs)
        idfList = [0] * len(self.vocabList)

        word_doc_count = Counter()
        for doc in docs:
            word_doc_count.update(set(doc))

        for word, count in tqdm(word_doc_count.items(), desc='计算IDF'):
            if word in self.vocabList:
                index = self.vocabList.index(word)
                idfList[index] = np.log(numDocs / (1 + count))

        return idfList

    def setWordToVec(self, docs):
        returnVec = []
        # 遍历文档中的所有单词
        for doc in docs:
            vec = [0] * len(self.vocabList)
            for word in doc:
                if word in self.vocabList:
                    vec[self.vocabList.index(word)] = 1
            returnVec.append(vec)
        return returnVec

    def bagWordToVec(self, docs):
        returnVec = []
        for doc in docs:
            vec = [0] * len(self.vocabList)
            for word in doc:
                if word in self.vocabList:
                    vec[self.vocabList.index(word)] += 1
            returnVec.append(vec)
        return returnVec

    def tfidfWordToVec(self, docs):
        """
        TF-IDF算法实现
        :param docs: List[List[str]] - 文档集合
        :return: List[List[float]] - 文档向量
        """
        rows = []
        for doc in docs:
            words = doc
            row = [0] * len(self.vocabList)
            word_count = {}
            for word in words:
                if word in self.vocabList:
                    word_count[word] = word_count.get(word, 0) + 1

            for word, count in word_count.items():
                if word in self.vocabList:
                    index = self.vocabList.index(word)
                    row[index] = count * self.idf[index]

            # row = self.mm(row)
            rows.append(row)
        return rows

    def mm(self, vec):
        """
        归一化向量 使得某一个特征对最终结果不会造成更大的影响
        :param vec: List[float] - 向量
        :return: List[float] - 归一化后的向量
        """
        minVal = min(vec)
        maxVal = max(vec)

        if maxVal == minVal:  # 避免除以0的情况
            return vec

        return [(x - minVal) / (maxVal - minVal) for x in vec]


class ParamSearchCV(object):
    def __init__(self, estimator, alphaList, cv=5, n_jobs=-1):
        self.estimator = estimator
        self.alphaList = alphaList
        self.cv = cv
        self.n_jobs = n_jobs
        self.best_params = None
        self.best_score = None
        self.cv_results = []

    def fit(self, X, y):
        self.best_score = -np.inf
        self.best_params = None

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.evaluate_params)(alpha, X, y) for alpha in tqdm(self.alphaList, desc="参数搜索")
        )

        for alpha, avg_score in results:
            self.cv_results.append({'alpha': alpha, 'score': avg_score})
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_params = alpha

    def evaluate_params(self, alpha, X, y):
        X = np.array(X)
        y = np.array(y)

        scores = []
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        for train_index, val_index in skf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            model = self.estimator.setAlpha(alpha)
            model.fit(X_train, y_train)
            scores.append(model.score(X_val, y_val))
        avg_score = np.mean(scores)
        return alpha, avg_score


def loadDataSet(target_spam_count=None):
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

    return docs, label


def downsample(docs, labels, target_spam_count):
    """
    下采样以平衡数据集
    :param docs: 文档集合
    :param labels: 类别标签
    :param target_spam_count: 目标垃圾短信数量
    :return: 下采样后的文档集合和类别标签
    """
    ham_docs = [doc for doc, label in zip(docs, labels) if label == 0]
    spam_docs = [doc for doc, label in zip(docs, labels) if label == 1]

    # 获取垃圾短信和非垃圾短信的实际数量
    spam_count = len(spam_docs)
    ham_count = len(ham_docs)

    # 如果目标垃圾短信数量超过实际数量，抛出异常
    if target_spam_count > spam_count:
        raise ValueError(f"目标垃圾短信数量 {target_spam_count} 超过实际垃圾短信数量 {spam_count}")

    # 对垃圾短信进行下采样
    spam_docs = random.sample(spam_docs, target_spam_count)
    spam_labels = [1] * target_spam_count

    # 对非垃圾短信进行下采样，使数量与目标垃圾短信数量相同
    ham_docs = random.sample(ham_docs, target_spam_count)
    ham_labels = [0] * target_spam_count

    # 合并下采样后的数据
    docs_downsampled = ham_docs + spam_docs
    labels_downsampled = ham_labels + spam_labels

    # 打乱数据顺序
    combined = list(zip(docs_downsampled, labels_downsampled))
    random.shuffle(combined)
    docs_downsampled, labels_downsampled = zip(*combined)

    return list(docs_downsampled), list(labels_downsampled)


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
            returnVec[vocabList.index(word)] = 1
    return returnVec


def bagOfWords2VecMN(vocabList, inputSet):
    """ji
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
        vocab_dict = defaultdict(int)
        for i, word in enumerate(self.vocabList):
            vocab_dict[word] = i

        for doc in tqdm(self.docs, desc='计算TF'):
            word_count = len(doc)
            if word_count == 0:
                tfList.append([0] * len(self.vocabList))
                continue
            tfDoc = [0] * len(self.vocabList)
            word_freq = Counter(doc)
            for word, freq in word_freq.items():
                index = vocab_dict[word]
                tfDoc[index] = freq
            tfDoc = np.array(tfDoc) / word_count
            tfList.append(tfDoc.tolist())
        return tfList

    def calc_idf(self):
        """
        计算逆文档频率 IDF
        :return: idfDict: 包含每个单词的IDF值的词典
        """
        numDocs = len(self.docs)
        idfList = [0] * len(self.vocabList)

        word_doc_count = Counter()
        for doc in self.docs:
            word_doc_count.update(set(doc))

        for word, count in tqdm(word_doc_count.items(), desc='计算IDF'):
            if word in self.vocabList:
                index = self.vocabList.index(word)
                idfList[index] = np.log(numDocs / (1 + count))

        return idfList

    def calc_tfidf(self):
        """
        TF-IDF算法实现
        :return: returnVec: 文档向量
        """
        tfList = self.calc_tf()
        idfList = self.calc_idf()
        tfidfList = []

        idfArray = np.array(idfList)
        for tfDoc in tqdm(tfList, desc='计算TF-IDF'):
            tfArray = np.array(tfDoc)
            tfidfDoc = tfArray * idfArray
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


def trainNB0(trainMatrix, trainCategory, alpha=0.1):
    """
    朴素贝叶斯分类器训练函数
    :param alpha:
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
    p0Num = np.ones(numWords) + alpha
    p1Num = np.ones(numWords) + alpha
    p0Denom = numWords * alpha
    p1Denom = numWords * alpha

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
    预测
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


def evaluate_model(p0V, p1V, pAb, X_test_vec, y_test):
    """
    评估模型
    :param p0V: 非垃圾词汇的概率
    :param p1V: 垃圾词汇的概率
    :param pAb: 垃圾短信的概率
    :param X_test_vec: 测试样本
    :param y_test: 测试样本对应的标签
    :return:
    """
    y_pred = [classifyNB(vec, p0V, p1V, pAb) for vec in X_test_vec]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, precision, recall, f1, conf_matrix


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
