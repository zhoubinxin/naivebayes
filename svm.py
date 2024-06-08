from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from naiveBayes import *

docs, label = loadDataSet()

vocabList = createVocabList(docs)

trainMat = []
for postinDoc in tqdm(docs):
    trainMat.append(setOfWords2Vec(vocabList, postinDoc))

trainMat = np.array(trainMat)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(trainMat, label, test_size=0.2, random_state=42)

# 训练SVM模型
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# 模型评估
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
