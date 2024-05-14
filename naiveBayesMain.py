# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# from importlib import reload
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    import naiveBayes
    from numpy import *
    listOposts,listClasses = naiveBayes.loadDataSet()
    myVocabList = naiveBayes.createVocabList(listOposts)
    print(myVocabList)
    print(naiveBayes.setOfWords2Vec(myVocabList,listOposts[0]))
    print(naiveBayes.setOfWords2Vec(myVocabList,listOposts[3]))

    listOposts,listClasses = naiveBayes.loadDataSet()
    myVocabList = naiveBayes.createVocabList(listOposts)
    trainMat = []
    for postinDoc in listOposts:
        trainMat.append(naiveBayes.setOfWords2Vec(myVocabList,postinDoc))

    p0V,p1V,pAb = naiveBayes.trainNB0(trainMat,listClasses)
    print(pAb)
    print(p0V)
    print(p1V)

    naiveBayes.testingNB()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
