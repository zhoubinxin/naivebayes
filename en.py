# from naiveBayesCN import *
from naiveBayes import *


# 全网格参数搜索
def main():
    docs, label = loadDataSet()

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(docs, label, test_size=0.2, random_state=1)

    # 使用 SimpleCountVectorizer 或 SimpleTfidfVectorizer
    vectorizer = WordToVec()  # 可以切换为 SimpleCountVectorizer
    vectorizer.fit(X_train)
    # setWordToVec bagWordToVec tfidfWordToVec
    X_train_vec = vectorizer.tfidfWordToVec(X_train)
    X_test_vec = vectorizer.tfidfWordToVec(X_test)

    # vectorizer = TfidfVectorizer()
    # X_train_vec = vectorizer.fit_transform(X_train)
    # X_test_vec = vectorizer.transform(X_test)

    # 使用 SimpleGridSearchCV 进行超参数搜索
    alphaList = [2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 2.9, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
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
    print(f"F1值: {f1}")
    print(conf_matrix)

    # 输出结果
    with open('result/best_score.txt', 'w', encoding='utf-8') as file:
        file.write(f"最佳参数: {best_params}\n")
        file.write(f"准确率: {accuracy}\n")
        file.write(f"精确率: {precision}\n")
        file.write(f"召回率: {recall}\n")
        file.write(f"F1值: {f1}\n")


if __name__ == '__main__':
    main()
