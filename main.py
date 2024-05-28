import json

import pandas as pd
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
    # idfDict = nb.computeIDF(listOposts)
    for postinDoc in tqdm(listOposts, desc='构建词向量矩阵'):
        trainMat.append(nb.setOfWords2Vec(myVocabList, postinDoc))
        # trainMat.append(nb.bagOfWords2VecMN(myVocabList, postinDoc))
        # trainMat.append(nb.bagOfWords2VecTFIDF(myVocabList, postinDoc, idfDict))
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

    print(f"准确率: {accuracy}")
    print(f"精确率: {precision}")
    print(f"召回率: {recall}")
    print(f"F1值: {f1}")

    # 构建json结果
    result = {
        "vocabList": myVocabList,
        "p0V": p0V.tolist(),
        "p1V": p1V.tolist(),
        "pAb": pAb,
    }
    # 保存结果到json文件
    with open('result/result.json', 'w', encoding='utf-8') as file:
        json.dump(result, file, ensure_ascii=False)

    # 保存数据到txt
    with open('result/score.txt', 'w', encoding='utf-8') as file:
        # 评估指标
        # accuracy, precision, recall, f1
        file.write("\n评估指标:\n")
        file.write(f'accuracy: {accuracy}\n')
        file.write(f'precision: {precision}\n')
        file.write(f'recall: {recall}\n')
        file.write(f'f1: {f1}\n')


if __name__ == '__main__':
    main()
