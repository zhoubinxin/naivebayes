from naiveBayes import loadDataSet, ParamSearchCV, NaiveBayes
from naiveBayesCN import *
from joblib import Parallel, delayed
import time
from itertools import islice
from tqdm import tqdm
import jieba
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef


# # 对半网格参数搜索
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

    start0 = time.perf_counter()
    filename = "scu_stopwords"
    stop_words = load_stop_words(filename)

    lines = 4000  # 数据量
    # sample_size = 10000  # 类别样本数量
    listOposts, listClasses = loadCNDataSet(lines, stop_words)

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
    param_grid = {'alpha': [3.0, 4.0, 5.0]}
    grid_search = SimpleGridSearchCV(SimpleNaiveBayes(), param_grid, cv=5)
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





# # Bert预处理+Logistic 回归
#
# # 加载预训练的BERT模型和分词器
# local_model_path = 'D:/桌面/R语言-数据挖掘-社会化网络/数据挖掘/naivebayes/bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(local_model_path)
# bert_model = BertModel.from_pretrained(local_model_path)
#
# def get_bert_embeddings(token_ids):
#     with torch.no_grad():
#         outputs = bert_model(torch.tensor(token_ids).unsqueeze(0))
#         return outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()  # 取平均作为句子向量
#
# def main():
#     start0 = time.perf_counter()
#     filename = "scu_stopwords"
#     stop_words = load_stop_words(filename)
#
#     lines = 5000
#     listOposts, listClasses = loadCNBertDataSet(lines, stop_words)
#
#     end1 = time.perf_counter()
#     runTime1 = end1 - start0
#     print("数据处理时间：", runTime1, "秒")
#
#     X_train, X_test, y_train, y_test = train_test_split(listOposts, listClasses, test_size=0.2, random_state=1)
#
#     # 转换训练和测试数据
#     print("转换训练数据：")
#     X_train_vec = [get_bert_embeddings(ids) for ids in tqdm(X_train, desc='训练数据')]
#
#     print("\n转换测试数据：")
#     X_test_vec = [get_bert_embeddings(ids) for ids in tqdm(X_test, desc='测试数据')]
#
#     start2 = time.perf_counter()
#
#     # 使用 Logistic 回归
#     model = LogisticRegression(max_iter=1000)#最多迭代一千次
#     print("训练模型：")
#     model.fit(X_train_vec, y_train)  # 不使用tqdm包装
#
#     end2 = time.perf_counter()
#     runTime2 = end2 - start2
#     print("模型训练时间：", runTime2, "秒")
#
#     start3 = time.perf_counter()
#
#     # 预测
#     print("预测数据：")
#     y_pred = model.predict(X_test_vec)
#
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, zero_division=1)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     mcc = matthews_corrcoef(y_test, y_pred)
#
#     end3 = time.perf_counter()
#     runTime3 = end3 - start3
#     print("预测时间：", runTime3, "秒")
#
#     cm = confusion_matrix(y_test, y_pred)
#
#     print(f"准确率: {accuracy}")
#     print(f"精确率: {precision}")
#     print(f"召回率: {recall}")
#     print(f"F1值: {f1}")
#     print(f"MCC: {mcc}")
#     print("混淆矩阵:")
#     print(cm)
#
#     with open('result/best_score.txt', 'w', encoding='utf-8') as file:
#         file.write(f"准确率: {accuracy}\n")
#         file.write(f"精确率: {precision}\n")
#         file.write(f"召回率: {recall}\n")
#         file.write(f"F1值: {f1}\n")
#         file.write(f"MCC: {mcc}\n")
#         file.write("混淆矩阵:\n")
#         file.write(np.array2string(cm))
#
#     end0 = time.perf_counter()
#     runTime0 = end0 - start0
#     print("总运行时间：", runTime0, "秒")
#
# if __name__ == '__main__':
#     main()


#Bert预处理+朴素贝叶斯

# # 加载预训练的BERT模型和分词器
# local_model_path = 'D:/桌面/R语言-数据挖掘-社会化网络/数据挖掘/naivebayes/bert-base-uncased'
# tokenizer = BertTokenizer.from_pretrained(local_model_path)
# bert_model = BertModel.from_pretrained(local_model_path)
#
# def get_bert_embeddings(token_ids):
#     with torch.no_grad():
#         outputs = bert_model(torch.tensor(token_ids).unsqueeze(0))
#         return outputs.last_hidden_state.squeeze(0).mean(dim=0).numpy()  # 取平均作为句子向量

# def main():
#     # 加载数据
#     docs, label = loadBertDataSet()
#
#     # 分割训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(docs, label, test_size=0.2, random_state=1)
#
#     # 初始化 tokenizer
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
#     # 确保所有文本都是字符串，并将列表连接成字符串
#     X_train = [' '.join(text) for text in X_train if isinstance(text, list) and text]
#     X_train_vec = [get_bert_embeddings(tokenizer.encode(text, add_special_tokens=True)) for text in tqdm(X_train, desc='训练数据')]
#
#     X_test = [' '.join(text) for text in X_test if isinstance(text, list) and text]
#     X_test_vec = [get_bert_embeddings(tokenizer.encode(text, add_special_tokens=True)) for text in tqdm(X_test, desc='测试数据')]
#
#     X_train_vec = np.array(X_train_vec)
#     X_test_vec = np.array(X_test_vec)
#
#     param_grid = {'alpha': [0.1, 1, 2, 1.5]}
#     grid_search = SimpleBertGridSearchCV(BertNaiveBayes(), param_grid, cv=5)
#     grid_search.fit(X_train_vec, y_train)
#
#     best_params = grid_search.best_params_
#
#     best_model = BertNaiveBayes(**best_params)
#     best_model.fit(X_train_vec, y_train)
#
#     y_pred = best_model.predict(X_test_vec)
#
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
#     recall = recall_score(y_test, y_pred, average='weighted')
#     f1 = f1_score(y_test, y_pred, average='weighted')
#     cm = confusion_matrix(y_test, y_pred)
#
#     print(f"准确率: {accuracy}")
#     print(f"精确率: {precision}")
#     print(f"召回率: {recall}")
#     print(f"F1值: {f1}")
#     print("混淆矩阵:")
#     print(cm)
#
#     os.makedirs('result', exist_ok=True)
#     with open('result/best_score.txt', 'w', encoding='utf-8') as file:
#         file.write(f"最佳参数: {best_params}\n")
#         file.write(f"准确率: {accuracy}\n")
#         file.write(f"精确率: {precision}\n")
#         file.write(f"召回率: {recall}\n")
#         file.write(f"F1值: {f1}\n")
#         file.write("混淆矩阵:\n")
#         file.write(np.array2string(cm))
#
# if __name__ == '__main__':
#     main()

# def main():
#     # 加载数据
#     docs, label = loadDataSet()
#
#     # 分割训练集和测试集
#     X_train, X_test, y_train, y_test = train_test_split(docs, label, test_size=0.2, random_state=1)
#
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
#     # 确保所有文本都是字符串，并过滤掉空字符串
#     print("转换训练数据：")
#     X_train = [str(text) for text in X_train]
#     X_train_vec = [get_bert_embeddings(tokenizer.encode(text, add_special_tokens=True)) for text in
#                    tqdm(X_train, desc='训练数据') if text.strip()]
#
#     print("转换测试数据：")
#     X_test = [str(text) for text in X_test]
#     X_test_vec = [get_bert_embeddings(tokenizer.encode(text, add_special_tokens=True)) for text in
#                   tqdm(X_test, desc='测试数据') if text.strip()]
#
#     X_train_vec = np.array(X_train_vec)
#     X_test_vec = np.array(X_test_vec)
#
#     # 使用 SimpleGridSearchCV 进行超参数搜索
#     alphaList = [1, 1.5, 2, 2.2, 2.3, 2.4, 2.5]
#     grid_search = ParamSearchCV(BertNaiveBayes(), alphaList, cv=5)
#     grid_search.fit(X_train_vec, y_train)
#     # 得到最佳参数
#     best_params = grid_search.best_params
#     print(f"最佳参数: {best_params}")
#
#     # 使用最佳参数训练模型
#     best_model = BertNaiveBayes(best_params)
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
#     conf_matrix = confusion_matrix(y_test, y_pred)
#
#     print(f"准确率: {accuracy}")
#     print(f"精确率: {precision}")
#     print(f"召回率: {recall}")
#     print(f"F1 值: {f1}")
#     print(f"混淆矩阵\n{conf_matrix}")
#
#     # 输出结果
#     with open('result/best_score.txt', 'w', encoding='utf-8') as file:
#         file.write(f"最佳参数: {best_params}\n")
#         file.write(f"准确率: {accuracy}\n")
#         file.write(f"精确率: {precision}\n")
#         file.write(f"召回率: {recall}\n")
#         file.write(f"F1值: {f1}\n")
#
#
#
# if __name__ == '__main__':
#     main()
