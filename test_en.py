import time
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from naiveBayesCN import *
import naiveBayes as nb


# 全网格参数搜索
def main():
    listOposts, listClasses = nb.loadDataSet()
    # vocabList = nb.createVocabList(listOposts)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(listOposts, listClasses, test_size=0.2, random_state=1)

    # # 使用 SimpleCountVectorizer 或 SimpleTfidfVectorizer
    vectorizer = SimpleCountVectorizer()  # 可以切换为 SimpleCountVectorizer
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 使用 SimpleGridSearchCV 进行超参数搜索
    param_grid = {'alpha': [2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]}
    grid_search = SimpleGridSearchCV(SimpleNaiveBayes(), param_grid, cv=len(param_grid['alpha']))
    grid_search.fit(X_train_vec, y_train)
    # 得到最佳参数
    best_params = grid_search.best_params_
    print(f"最佳参数: {best_params}")

    # 使用最佳参数训练模型
    # best_model = SimpleNaiveBayes(3.0)
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
        file.write(f"最佳参数: {best_params}\n")
        file.write(f"准确率: {accuracy}\n")
        file.write(f"精确率: {precision}\n")
        file.write(f"召回率: {recall}\n")
        file.write(f"F1值: {f1}\n")


if __name__ == '__main__':
    main()
