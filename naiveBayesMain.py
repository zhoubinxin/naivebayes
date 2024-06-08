from mlxtend.evaluate import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from naiveBayesv1 import *
from numpy import *


def main():
    # 加载数据集
    listOposts, listClasses = loadDataSet()

    # 创建词汇表
    myVocabList = createVocabList(listOposts)

    # 训练朴素贝叶斯分类器
    trainMat = []
    for postinDoc in tqdm(listOposts):
        # trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
        trainMat.append(bagOfWords2VecMN(myVocabList, postinDoc))

    X_train, X_test, y_train, y_test = train_test_split(trainMat, listClasses, test_size=0.2, random_state=1)
    # 训练数据
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))

    # 测试数据
    y_pred = [classifyNB(vec, p0V, p1V, pAb) for vec in X_test]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"准确率: {accuracy}")
    print(f"精确率: {precision}")
    print(f"召回率: {recall}")
    print(f"F1 值: {f1}")
    print(f"混淆矩阵\n{conf_matrix}")


if __name__ == '__main__':
    main()
