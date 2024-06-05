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