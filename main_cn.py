import json

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing as mp
from itertools import repeat
from naiveBayesCN import *
import naiveBayes as nb

def main():
    stop_words = load_stop_words()

    # 数据下采样
    lines = 5000  # 数据量
    listOposts, listClasses = loadDataSet(stop_words, lines)

    # 并行预处理文档
    with mp.Pool(mp.cpu_count()) as pool:
        preprocessed_docs = list(tqdm(pool.imap(preprocess_doc, zip(listOposts, repeat(stop_words))),
                                      total=len(listOposts), desc='预处理文档'))

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_docs, listClasses, test_size=0.2, random_state=1)

    # 使用 SimpleCountVectorizer 或 SimpleTfidfVectorizer
    vectorizer = SimpleCountVectorizer()  # 可以切换为 SimpleCountVectorizer
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 使用 SimpleGridSearchCV 进行超参数搜索
    # param_grid = {'alpha': [0.7, 1.3]}
    # while True:
    #     alpha = int(input("请输入alpha的值："))
    #     grid_search = SimpleGridSearchCV(SimpleNaiveBayes(), alpha, cv=3)
    #     grid_search.fit(X_train_vec, y_train)

    # 网格搜索
    min_alpha = 2.1
    max_alpha = 2.2
    best_f1 = 0
    best_alpha = None
    best_metrics = None

    for alpha in np.arange(min_alpha, max_alpha, 0.1):
        p0V, p1V, pAb = nb.trainNB0(X_train, y_train, alpha)
        accuracy, precision, recall, f1, conf_matrix = nb.evaluate_model(p0V, p1V, pAb, X_test, y_test)

        print(f"Alpha: {alpha}")
        print(f"准确率Accuracy: {accuracy}")
        print(f"精确率Precision: {precision}")
        print(f"召回率Recall: {recall}")
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
    p0V, p1V, pAb = nb.trainNB0(X_train_vec, listClasses, best_alpha)

    result = {
        "vocabList": preprocessed_docs,
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

    # 得到最佳参数
    # best_params = grid_search.best_params_
    # print(f"最佳参数: {best_params}")
    #
    # # 使用最佳参数训练模型
    # best_model = SimpleNaiveBayes(**best_params)
    # best_model.fit(X_train_vec, y_train)
    #
    # # 预测
    # y_pred = best_model.predict(X_test_vec)
    #
    # # 评估模型
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    # recall = recall_score(y_test, y_pred, average='weighted')
    # f1 = f1_score(y_test, y_pred, average='weighted')
    #
    # print(f"准确率: {accuracy}")
    # print(f"精确率: {precision}")
    # print(f"召回率: {recall}")
    # print(f"F1值: {f1}")
    #
    # # 输出结果
    # with open('result/best_score.txt', 'w', encoding='utf-8') as file:
    #     file.write(f"最佳参数: {best_params}\n")
    #     file.write(f"准确率: {accuracy}\n")
    #     file.write(f"精确率: {precision}\n")
    #     file.write(f"召回率: {recall}\n")
    #     file.write(f"F1值: {f1}\n")


if __name__ == '__main__':
    main()
