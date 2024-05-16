from tqdm import tqdm

import naiveBayes
from numpy import *


def main():
    # 加载数据集
    listOposts, listClasses = naiveBayes.loadDataSet()

    # 创建词汇表
    myVocabList = naiveBayes.createVocabList(listOposts)

    # 训练朴素贝叶斯分类器
    trainMat = []
    for postinDoc in tqdm(listOposts):
        trainMat.append(naiveBayes.setOfWords2Vec(myVocabList, postinDoc))

    # 训练数据
    p0V, p1V, pAb = naiveBayes.trainNB0(array(trainMat), array(listClasses))

    print(p0V, p1V, pAb)


if __name__ == '__main__':
    main()
