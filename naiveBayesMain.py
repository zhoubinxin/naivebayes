from tqdm import tqdm

import naiveBayes
from numpy import *


def main():
    # 读取数据
    listOposts, listClasses = naiveBayes.loadDataSet()

    myVocabList = naiveBayes.createVocabList(listOposts)
    print(myVocabList)
    print(naiveBayes.setOfWords2Vec(myVocabList, listOposts[0]))
    print(naiveBayes.setOfWords2Vec(myVocabList, listOposts[3]))

    listOposts, listClasses = naiveBayes.loadDataSet()
    myVocabList = naiveBayes.createVocabList(listOposts)
    trainMat = []
    for postinDoc in tqdm(listOposts):
        trainMat.append(naiveBayes.setOfWords2Vec(myVocabList, postinDoc))
    print(trainMat)
    p0V, p1V, pAb = naiveBayes.trainNB0(trainMat, listClasses)
    print(pAb)
    print(p0V)
    print(p1V)

    naiveBayes.testingNB()


if __name__ == '__main__':
    main()
