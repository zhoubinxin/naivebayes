import json

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import naiveBayes as nb
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def main():
    # 加载数据集
    docs, label = nb.loadDataSet()
    # 创建词汇表
    vocabList = nb.createVocabList(docs)

    # 构建词向量矩阵
    trainMat = []
    # for inputSet in tqdm(docs, desc='构建词向量矩阵'):
    # trainMat.append(nb.setOfWords2Vec(vocabList, inputSet))
    # trainMat.append(nb.bagOfWords2VecMN(vocabList, inputSet))
    tfidf = nb.TFIDF(docs, vocabList)
    trainMat = tfidf.calc_tfidf()

    # 将数据集划分为训练集和测试集
    # test_size 表示测试集的比例
    # random_state 表示随机数的种子，保证每次划分的数据集都是相同的
    X_train, X_test, y_train, y_test = train_test_split(trainMat, label, test_size=0.2, random_state=1)

    # 网格搜索
    min_alpha = 10
    max_alpha = 15
    best_f1 = 0
    best_alpha = None
    best_metrics = None

    for alpha in np.arange(min_alpha, max_alpha, 1):
        p0V, p1V, pAb = nb.trainNB0(X_train, y_train, alpha)
        accuracy, precision, recall, f1, conf_matrix = nb.evaluate_model(p0V, p1V, pAb, X_test, y_test)

        print(f"Alpha: {alpha}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Confusion Matrix:\n{conf_matrix}\n")

        if f1 > best_f1:
            best_f1 = f1
            best_alpha = alpha
            best_metrics = (accuracy, precision, recall, f1, conf_matrix)

    print(f"Best Alpha: {best_alpha}")
    print(f"""
    Best Metrics
    Accuracy: {best_metrics[0]}
    Precision: {best_metrics[1]}
    Recall: {best_metrics[2]}
    F1 Score: {best_metrics[3]}
    Confusion Matrix: \n{best_metrics[4]}
    """)

    # 训练最佳模型
    p0V, p1V, pAb = nb.trainNB0(trainMat, label, best_alpha)

    result = {
        "vocabList": vocabList,
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
        file.write(f'accuracy: {best_metrics[0]}\n')
        file.write(f'precision: {best_metrics[1]}\n')
        file.write(f'recall: {best_metrics[2]}\n')
        file.write(f'f1: {best_metrics[3]}\n')

        # 混淆矩阵
        file.write("\nconf_matrix:\n")
        file.write(str(best_metrics[4]) + '\n')

    # 创建词云对象
    # wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(vocabList)

    # 显示词云
    # plt.figure()
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis("off")
    # plt.show()


if __name__ == '__main__':
    main()
