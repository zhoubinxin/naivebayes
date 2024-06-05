import numpy as np
import time
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing as mp
from itertools import repeat

from naiveBayesCN import *


def main():
    start0 = time.perf_counter()
    stop_words = load_stop_words("scu_stopwords")

    lines = 30000  # 数据量
    listOposts, listClasses = loadCNDataSet(lines, stop_words)

    end1 = time.perf_counter()
    runTime1 = end1 - start0
    print("数据处理时间：", runTime1, "秒")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(listOposts, listClasses, test_size=0.2, random_state=1)

    # 使用 SimpleCountVectorizer 或 SimpleTfidfVectorizer
    vectorizer = SimpleCountVectorizer()  # 可以切换为 SimpleCountVectorizer
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    start2 = time.perf_counter()

    # 定义参数网格
    param_grid = {'alpha': [6.3, 6.5, 6.7]}

    # 使用 SimpleHalvingGridSearchCV 进行超参数搜索
    halving_grid_search = SimpleHalvingGridSearchCV(SimpleNaiveBayes(), param_grid, cv=len(param_grid['alpha']))
    halving_grid_search.fit(X_train_vec, y_train)

    # 得到最佳参数
    best_params = halving_grid_search.best_params_
    if best_params is None:
        print("未找到最佳参数,停止训练")
        return

    print(f"最佳参数: {best_params}")

    end2 = time.perf_counter()
    runTime2 = end2 - start2
    print("超参数搜索时间：", runTime2, "秒")

    # 使用最佳参数训练模型
    best_model = SimpleNaiveBayes(**best_params)
    best_model.fit(X_train_vec, y_train)

    # 预测
    y_pred = best_model.predict(X_test_vec)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"准确率: {accuracy}")
    print(f"精确率: {precision}")
    print(f"召回率: {recall}")
    print(f"F1值: {f1}")

    # 输出结果
    with open('result/best_score.txt', 'w', encoding='utf-8') as file:
        file.write(f"准确率: {accuracy}\n")
        file.write(f"精确率: {precision}\n")
        file.write(f"召回率: {recall}\n")
        file.write(f"F1值: {f1}\n")

    end0 = time.perf_counter()
    runTime0 = end0 - start0
    print("运行时间：", runTime0, "秒")


if __name__ == '__main__':
    main()

# #对半网格参数搜索
# def main():
#     start0 = time.perf_counter()
#     filename="scu_stopwords"
#     stop_words = load_stop_words(filename)
#
#     lines = 30000  # 数据量
#     listOposts, listClasses = loadCNDataSet(lines, stop_words)
#
#
#     end1 = time.perf_counter()
#     runTime1 = end1 - start0
#     print("数据处理时间：",runTime1,"秒")
#
#     # 划分数据集
#     X_train, X_test, y_train, y_test = train_test_split(listOposts, listClasses, test_size=0.2, random_state=1)
#
#     # 使用 SimpleCountVectorizer 或 SimpleTfidfVectorizer
#     vectorizer = SimpleCountVectorizer()  # 可以切换为 SimpleCountVectorizer
#     X_train_vec = vectorizer.fit_transform(X_train)
#     X_test_vec = vectorizer.transform(X_test)
#
#     start2 = time.perf_counter()
#
#     # 使用 SimpleGridSearchCV 进行超参数搜索
#     param_grid = {'alpha': [5.0, 6.1]}
#     grid_search = SimpleGridSearchCV(SimpleNaiveBayes(), param_grid, cv=3)
#     grid_search.fit(X_train_vec, y_train)
#
#     # 得到最佳参数
#     best_params = grid_search.best_params_
#     print(f"最佳参数: {best_params}")
#
#     end2 = time.perf_counter()
#     runTime2 = end2 - start2
#     print("超参数搜索时间：", runTime2, "秒")
#
#     # 使用最佳参数训练模型
#     # best_model = SimpleNaiveBayes(3.0)
#     best_model = SimpleNaiveBayes(**best_params)
#     best_model.fit(X_train_vec, y_train)
#
#     # 预测
#     y_pred = best_model.predict(X_test_vec)
#
#     # 评估模型
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
#
#     print(f"准确率: {accuracy}")
#     print(f"精确率: {precision}")
#     print(f"召回率: {recall}")
#     print(f"F1值: {f1}")
#
#     # 输出结果
#     with open('result/best_score.txt', 'w', encoding='utf-8') as file:
#         # file.write(f"最佳参数: {best_params}\n")
#         file.write(f"准确率: {accuracy}\n")
#         file.write(f"精确率: {precision}\n")
#         file.write(f"召回率: {recall}\n")
#         file.write(f"F1值: {f1}\n")
#
#     end0 = time.perf_counter()
#     runTime0 = end0 - start0
#     print("运行时间：",runTime0,"秒")
#
# if __name__ == '__main__':
#     main()
