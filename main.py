from mlxtend.evaluate import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import naiveBayes as nb


def main():
    # 加载数据集
    listOposts, listClasses = nb.loadDataSet()

    # 创建词汇表
    myVocabList = nb.createVocabList(listOposts)

    # 构建词向量矩阵
    trainMat = []
    for postinDoc in tqdm(listOposts):
        trainMat.append(nb.setOfWords2Vec(myVocabList, postinDoc))

    # 将数据集划分为训练集和测试集
    # test_size 表示测试集的比例
    # random_state 表示随机数的种子，保证每次划分的数据集都是相同的
    X_train, X_test, y_train, y_test = train_test_split(trainMat, listClasses, test_size=0.2, random_state=1)

    # 训练朴素贝叶斯分类器
    p0V, p1V, pAb = nb.trainNB0(X_train, y_train)

    # 预测测试集
    y_pred = [nb.classifyNB(doc, p0V, p1V, pAb) for doc in X_test]

    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


if __name__ == '__main__':
    main()
