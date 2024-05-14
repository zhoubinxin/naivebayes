import naiveBayes
from numpy import *

if __name__ == '__main__':
    listOposts, listClasses = naiveBayes.loadDataSet()
    myVocabList = naiveBayes.createVocabList(listOposts)
    print(myVocabList)
    print(naiveBayes.setOfWords2Vec(myVocabList, listOposts[0]))
    print(naiveBayes.setOfWords2Vec(myVocabList, listOposts[3]))

    listOposts, listClasses = naiveBayes.loadDataSet()
    myVocabList = naiveBayes.createVocabList(listOposts)
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(naiveBayes.setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = naiveBayes.trainNB0(trainMat, listClasses)
    print(pAb)
    print(p0V)
    print(p1V)

    naiveBayes.testingNB()
