from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
import matplotlib as mpl

from naiveBayes import *


def main():
    docs, label = loadDataSet()

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(docs, label, test_size=0.2, random_state=1)

    vectorizer = WordToVec()
    vectorizer.fit_tfidf(docs)
    # setWordToVec bagWordToVec tfidfWordToVec
    X_train_vec = vectorizer.tfidfWordToVec(X_train)
    X_test_vec = vectorizer.tfidfWordToVec(X_test)

    # 使用 SimpleGridSearchCV 进行超参数搜索
    alphaList = [1, 1.5, 2, 2.2, 2.3, 2.4, 2.5]
    grid_search = ParamSearchCV(NaiveBayes(), alphaList, cv=5)
    grid_search.fit(X_train_vec, y_train)
    # 得到最佳参数
    best_params = grid_search.best_params
    print(f"最佳参数: {best_params}")

    # 使用最佳参数训练模型
    # best_model = SimpleNaiveBayes(3.0)
    best_model = NaiveBayes(best_params)
    best_model.fit(X_train_vec, y_train)

    # 预测
    y_pred = best_model.predict(X_test_vec)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"准确率: {accuracy}")
    print(f"精确率: {precision}")
    print(f"召回率: {recall}")
    print(f"F1 值: {f1}")
    print(f"混淆矩阵\n{conf_matrix}")

    # 输出结果
    with open('result/best_score.txt', 'w', encoding='utf-8') as file:
        file.write(f"最佳参数: {best_params}\n")
        file.write(f"准确率: {accuracy}\n")
        file.write(f"精确率: {precision}\n")
        file.write(f"召回率: {recall}\n")
        file.write(f"F1值: {f1}\n")

    # 设置中文字体
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

    # 计算Precision-Recall曲线和平均精确率
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    average_precision = average_precision_score(y_test, y_pred)

    # 绘制Precision-Recall曲线
    plt.figure()
    plt.step(recall, precision, where='post', color='b', alpha=0.7,
             label='Precision-Recall曲线 (AP = %0.2f)' % average_precision)
    plt.fill_between(recall, precision, step='post', alpha=0.3, color='b')

    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall 曲线')
    plt.legend(loc="lower left")
    plt.show()


if __name__ == '__main__':
    main()
