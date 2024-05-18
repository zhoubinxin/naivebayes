import re
from itertools import islice

import jieba
from mlxtend.evaluate import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


def loadDataSet(lines=800000):
    """
    读取中文数据集

    :return:
    """
    postingList = []  # 存储文本
    classVec = []  # 存储标签

    with open('./data/cnsmss/80w.txt', 'r', encoding='utf-8') as file:
        dataSet = [line.strip().split('\t') for line in islice(file, lines)]

    for item in tqdm(dataSet, desc='加载数据集：'):
        # 0：非垃圾短信；1：垃圾短信
        classVec.append(int(item[1]))

        # 将每条短信拆分为单词列表
        try:
            postingList.append(item[2])
        except IndexError:
            postingList.append(' ')
            print(item)
            # 空文本
            pass

        # try:
        #     words = jieba.lcut(item[2], cut_all=False)
        #     # 去除停用词
        #     for word in words:
        #         if word in stop_words:
        #             words.remove(word)
        #     postingList.append(words)
        # except IndexError:
        #     # 空文本
        #     pass

    return postingList, classVec


texts, labels = loadDataSet()

# 文本向量化
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)
print(X)
# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=1)

# 初始化模型
model = MultinomialNB()

# 训练模型
model.fit(X_train, y_train)

# 进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('准确率：', accuracy)
print('精确率：', precision)
print('召回率：', recall)
print('F1值：', f1)
