
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from cncn import *

# 全网格参数搜索
def main():

    stop_words = load_stop_words("scu_stopwords")

    lines = 10000  # 数据量
    listOposts, listClasses = loadCNDataSet(lines, stop_words)


    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(listOposts, listClasses, test_size=0.2, random_state=1)

    # 使用 SimpleCountVectorizer 或 SimpleTfidfVectorizer
    vectorizer = SimpleTfidfVectorizer()  # 可以切换为 SimpleCountVectorizer
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 使用 SimpleGridSearchCV 进行超参数搜索
    param_grid = {
        'alpha': [400.0, 4.0],
        # 'vectorizer__max_df': [0.75, 1.0],  # 针对 SimpleCountVectorizer，aplha最优：4.0
        'vectorizer__min_df': [0.1, 0.05],  # 针对 SimpleCountVectorizer和SimpleTfidfVectorizer
        'vectorizer__use_idf': [True, False]  # 针对 SimpleTfidfVectorizer，aplha最优：400
    }
    grid_search = SimpleGridSearchCV(SimpleNaiveBayes(), param_grid, cv=3)
    grid_search.fit(X_train_vec, y_train)

    # 获取最佳参数
    best_params = grid_search.best_params_
    print(f"最佳参数: {best_params}")

    # 提取与 SimpleNaiveBayes 相关的参数
    nb_params = {k: v for k, v in best_params.items() if k in ["alpha"]}

    # 使用最佳参数重新初始化 SimpleTfidfVectorizer 或 SimpleCountVectorizer
    vectorizer_params = {k.split("__")[1]: v for k, v in best_params.items() if k.startswith("vectorizer__")}
    vectorizer.set_params(**vectorizer_params)

    # 使用最佳参数训练模型
    best_model = SimpleNaiveBayes(**nb_params)
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


if __name__ == '__main__':
    main()

