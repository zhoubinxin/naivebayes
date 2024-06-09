from naiveBayesCN import *
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib
from tkinter import Tk


# def main():
#     start0 = time.perf_counter()
#     stop_words = load_stop_words("scu_stopwords")
#
#     lines = 10000  # 数据量
#     sample_size = 10000  # 类别样本数量
#     listOposts, listClasses = loadCNDataSet(lines, stop_words,sample_size)
#
#     end1 = time.perf_counter()
#     runTime1 = end1 - start0
#     print("数据处理时间：", runTime1, "秒")
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
#     # 定义参数网格
#     param_grid = {'alpha': [0.3, 0.1]}
#
#     # 使用 SimpleHalvingGridSearchCV 进行超参数搜索
#     halving_grid_search = SimpleHalvingGridSearchCV(SimpleNaiveBayes(), param_grid, cv=len(param_grid['alpha']))
#     halving_grid_search.fit(X_train_vec, y_train)
#
#     # 得到最佳参数
#     best_params = halving_grid_search.best_params_
#     if best_params is None:
#         print("未找到最佳参数，退出程序。")
#         return
#
#     print(f"最佳参数: {best_params}")
#
#     end2 = time.perf_counter()
#     runTime2 = end2 - start2
#     print("超参数搜索时间：", runTime2, "秒")
#
#     # 使用最佳参数训练模型
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
#         file.write(f"准确率: {accuracy}\n")
#         file.write(f"精确率: {precision}\n")
#         file.write(f"召回率: {recall}\n")
#         file.write(f"F1值: {f1}\n")
#
#     end0 = time.perf_counter()
#     runTime0 = end0 - start0
#     print("运行时间：", runTime0, "秒")
#
#
# if __name__ == '__main__':
#     main()



# 全网格参数搜索
def main():

    # # 设置中文字体
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    start0 = time.perf_counter()
    filename = "scu_stopwords"
    stop_words = load_stop_words(filename)

    lines = 150000  # 数据量
    sample_size = 150000  # 类别样本数量
    listOposts, listClasses = loadCNDataSet(lines, stop_words, sample_size)

    end1 = time.perf_counter()
    runTime1 = end1 - start0
    print("数据处理时间：", runTime1, "秒")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(listOposts, listClasses, test_size=0.2, random_state=1)

    # 使用 SimpleCountVectorizer 或 SimpleTfidfVectorizer
    vectorizer = SimpleCountVectorizer()  # 可以切换为 SimpleTfidfVectorizer
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    start2 = time.perf_counter()

    # 使用 SimpleGridSearchCV 进行超参数搜索
    param_grid = {'alpha': [0.4, 0.5, 0.3]}
    grid_search = SimpleGridSearchCV(SimpleNaiveBayes(), param_grid, cv=15)
    grid_search.fit(X_train_vec, y_train)

    # 得到最佳参数
    best_params = grid_search.best_params_
    print(f"最佳参数: {best_params}")

    end2 = time.perf_counter()
    runTime2 = end2 - start2
    print("超参数搜索时间：", runTime2, "秒")

    start3 = time.perf_counter()

    # 使用最佳参数训练模型
    best_model = SimpleNaiveBayes(**best_params)
    best_model.fit(X_train_vec, y_train)

    # 预测
    y_pred = best_model.predict(X_test_vec)
    # y_pred_prob = best_model.predict_proba(X_test_vec)[:, 1]  # 是二分类问题

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    # auc = roc_auc_score(y_test, y_pred_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    end3 = time.perf_counter()
    runTime3 = end3 - start3
    print("模型训练时间：", runTime3, "秒")

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    print(f"准确率: {accuracy}")
    print(f"精确率: {precision}")
    print(f"召回率: {recall}")
    print(f"F1值: {f1}")
    # print(f"AUC: {auc}")
    print(f"MCC: {mcc}")
    print("混淆矩阵:")
    print(cm)

    # 输出结果
    with open('result/best_score.txt', 'w', encoding='utf-8') as file:
        file.write(f"最佳参数: {best_params}\n")
        file.write(f"准确率: {accuracy}\n")
        file.write(f"精确率: {precision}\n")
        file.write(f"召回率: {recall}\n")
        file.write(f"F1值: {f1}\n")
        # file.write(f"AUC: {auc}\n")
        file.write(f"MCC: {mcc}\n")
        file.write("混淆矩阵:\n")
        file.write(np.array2string(cm))

    # # 绘制ROC曲线
    # fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic (ROC) Curve')
    # plt.legend(loc="lower right")
    #
    # # 设置窗口标题
    # fig = plt.gcf()
    # fig.canvas.manager.set_window_title('ROC 曲线')
    #
    # plt.savefig('result/roc_curve.png')  # 保存绘图
    # plt.show()

    # 绘制KS曲线
    # plot_ks_curve(y_test, y_pred_prob)

    end0 = time.perf_counter()
    runTime0 = end0 - start0
    print("运行时间：", runTime0, "秒")

if __name__ == '__main__':
    # matplotlib.use('TkAgg')  # 使用 TkAgg 后端以独立窗口显示图形
    main()


# # 半朴素贝叶斯SPODE
# def main():
#     start0 = time.perf_counter()
#     filename="scu_stopwords"
#     stop_words = load_stop_words(filename)
#
#     lines = 5000  # 数据量
#     sample_size = 5000  # 类别样本数量
#     listOposts, listClasses = loadCNDataSet(lines, stop_words, sample_size)
#
#     end1 = time.perf_counter()
#     runTime1 = end1 - start0
#     print("数据处理时间：", runTime1, "秒")
#
#     # 划分数据集
#     X_train, X_test, y_train, y_test = train_test_split(listOposts, listClasses, test_size=0.2, random_state=1)
#
#     # 使用 SimpleCountVectorizer 或 SimpleTfidfVectorizer
#     vectorizer = SimpleTfidfVectorizer()  # 可以切换为 SimpleTfidfVectorizer
#     X_train_vec = vectorizer.fit_transform(X_train)
#     X_test_vec = vectorizer.transform(X_test)
#
#     start2 = time.perf_counter()
#
#     # 使用 SimpleGridSearchCV 进行超参数搜索
#     param_grid = {'alpha': [0.01, 0.05]}
#     grid_search = SimpleGridSearchCV(SimpleSPODE(), param_grid, cv=2)
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
#     best_model = SimpleSPODE(**best_params)
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
#         file.write(f"最佳参数: {best_params}\n")
#         file.write(f"准确率: {accuracy}\n")
#         file.write(f"精确率: {precision}\n")
#         file.write(f"召回率: {recall}\n")
#         file.write(f"F1值: {f1}\n")
#
#     end0 = time.perf_counter()
#     runTime0 = end0 - start0
#     print("运行时间：", runTime0, "秒")
#
# if __name__ == '__main__':
#     main()
