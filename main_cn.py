from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import nativeBayesCN as nbcn
import multiprocessing as mp
from itertools import repeat
from tqdm import tqdm

import pandas as pd

def main():
    # 加载停用词
    stop_words = nbcn.load_stop_words()

    # 加载数据集，并使用多进程预处理以加速
    lines = 10000  # 数据量可以根据实际情况调整
    listOposts, listClasses = nbcn.loadDataSet(stop_words, lines)

    # 多进程预处理文档
    with mp.Pool(mp.cpu_count()) as pool:
        preprocessed_docs = list(
            tqdm(pool.imap(nbcn.preprocess_doc, zip(listOposts, repeat(stop_words))),
                 total=len(listOposts), desc='预处理文档'))

    # 定义模型与参数网格
    param_grid = {
        'count_vect__max_df': (0.5, 0.75, 1.0),
        'gb_clf__learning_rate': (0.01, 0.1, 0.5),
        'gb_clf__n_estimators': (50, 100, 200),
    }
    pipeline = Pipeline([
        ('count_vect', CountVectorizer()),
        ('gb_clf', GradientBoostingClassifier(random_state=1))
    ])

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_docs, listClasses,
                                                        test_size=0.2, random_state=1)

    # 使用GridSearchCV进行模型调优
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    # 最佳参数
    best_params = grid_search.best_params_
    print(f"最佳参数: {best_params}")

    # 使用最佳参数的模型进行预测
    y_pred = grid_search.predict(X_test)

    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"准确率: {accuracy}")
    print(f"精确率: {precision}")
    print(f"召回率: {recall}")
    print(f"F1值: {f1}")

    # 保存结果到txt文件
    with open('result/best_score.txt', 'w', encoding='utf-8') as file:
        file.write(f"最佳参数: {best_params}\n")
        file.write(f"准确率: {accuracy}\n")
        file.write(f"精确率: {precision}\n")
        file.write(f"召回率: {recall}\n")
        file.write(f"F1值: {f1}\n")


if __name__ == '__main__':
    main()
