import naiveBayesCN as nbcn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import multiprocessing as mp
from itertools import repeat

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

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_docs, listClasses,
                                                        test_size=0.2, random_state=1)

    # 定义参数网格
    param_grid = {
        'max_df': [0.5, 0.75, 1.0],
        'alpha': [1.0, 0.1, 0.01]
    }

    # 手动实现的参数搜索
    best_params, best_score = nbcn.grid_search_naive_bayes(X_train, y_train, X_test, y_test, param_grid)

    print(f"最佳参数: {best_params}")
    print(f"最佳准确率: {best_score}")

    # 使用最佳参数的模型进行预测
    vectorizer = CountVectorizer(max_df=best_params['max_df'])
    X_train_vec = vectorizer.fit_transform(X_train).toarray()
    X_test_vec = vectorizer.transform(X_test).toarray()

    p0V, p1V, pAb = nbcn.trainNB0(X_train_vec, y_train)
    y_pred = [nbcn.classifyNB(vec, p0V, p1V, pAb) for vec in tqdm(X_test_vec, desc="分类测试集")]

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
