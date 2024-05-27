网页链接：
https://gitcode.csdn.net/65e935ed1a836825ed78d617.html?dp_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpZCI6MjE0MTgwOSwiZXhwIjoxNzE3NDMxNjAxLCJpYXQiOjE3MTY4MjY4MDEsInVzZXJuYW1lIjoicXFfNjYxMzgxMTMifQ.ZWG7riWIOqM59mkSJiDPp3yFqtpmrCY6x7pK6YqPFH4

CountVectorize
CountVectorizer是属于常见的特征数值计算类，是一个文本特征提取方法。对于每一个训练文本，它只考虑每种词汇在该训练文本中出现的频率。

CountVectorizer会将文本中的词语转换为词频矩阵，它通过fit_transform函数计算各个词语出现的次数。

CountVectorizer参数详解
```python
CountVectorizer(input='content', encoding='utf-8',  decode_error='strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, stop_words=None, 
token_pattern='(?u)\b\w\w+\b', ngram_range=(1, 1), analyzer='word', max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype=<class 'numpy.int64'>)
```
CountVectorizer类的参数很多，分为三个处理步骤：preprocessing、tokenizing、n-grams generation.

一般要设置的参数是:ngram_range,max_df，min_df，max_features等，具体情况具体分析

![img_1.png](img_1.png)
![img_2.png](img_2.png)
![img_3.png](img_3.png)
![img_4.png](img_4.png)

用数据输入形式为列表，列表元素为代表文章的字符串，一个字符串代表一篇文章，字符串是已经分割好的。CountVectorizer同样适用于中文;
CountVectorizer是通过fit_transform函数将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在第i个文本下的词频。即各个词语出现的次数，通过get_feature_names()可看到所有文本的关键字，通过toarray()可看到词频矩阵的结果。
入门示例
```python
from sklearn.feature_extraction.text import CountVectorizer

texts=["dog cat fish","dog cat cat","fish bird", 'bird'] # “dog cat fish” 为输入列表元素,即代表一个文章的字符串
cv = CountVectorizer()#创建词袋数据结构
cv_fit=cv.fit_transform(texts)
#上述代码等价于下面两行
#cv.fit(texts)
#cv_fit=cv.transform(texts)

print(cv.get_feature_names())    #['bird', 'cat', 'dog', 'fish'] 列表形式呈现文章生成的词典

print(cv.vocabulary_	)              # {‘dog’:2,'cat':1,'fish':3,'bird':0} 字典形式呈现，key：词，value:词频

print(cv_fit)
# （0,3） 1   第0个列表元素，**词典中索引为3的元素**， 词频
#（0,1）1
#（0,2）1
#（1,1）2
#（1,2）1
#（2,0）1
#（2,3）1
#（3,0）1

print(cv_fit.toarray()) #.toarray() 是将结果转化为稀疏矩阵矩阵的表示方式；
#[[0 1 1 1]
# [0 2 1 0]
# [1 0 0 1]
# [1 0 0 0]]

print(cv_fit.toarray().sum(axis=0))  #每个词在所有文档中的词频
#[2 3 2 2]
```
设置停用词列表，处理中文文档
```python
#从文件导入停用词表
stpwrdpath ="C:\\Users\\Administrator\Desktop\lect09_codes\lect09_proj\stop_words\\中文停用词库.txt"
with open(stpwrdpath, 'rb') as fp:
    stopword = fp.read().decode('utf-8')  # 提用词提取
#将停用词表转换为list  
stpwrdlst = stopword.splitlines()
# 从sklearn.feature_extraction.text里导入CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
# 对CountVectorizer进行初始化（去除中文停用词）
count_vec=CountVectorizer(stop_words=stpwrdlst) #创建词袋数据结构
X_count_train = count_vec.fit_transform(all_list[:2])  #<class 'scipy.sparse.csr.csr_matrix'>
# 将原始训练和测试文本转化为特征向量
X_count_train= X_count_train.toarray()
X_count_test = count_vec.transform(all_list[2]).toarray()
print(X_count_train)
#词汇表
print('\nvocabulary list:\n\n',count_vec.get_feature_names())
print( '\nvocabulary dic :\n\n',count_vec.vocabulary_)
print ('vocabulary:\n\n')
for key,value in count_vec.vocabulary_.items():
    print(key,value)
```
训练集也就是a，b 的词频统计结果，词汇列表、字典为：
```python
[[1 1 1 1 1 0 0 2 1 0 1 1 1 0 1 0 0 0 1 0 1 1 0 3 1 2 0 0 1 0 0 1 1 1 2]
 [0 0 0 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 0 1 3 0 1 4 0 0 1 1 1 1 1 0 1 1 1]]

vocabulary list:

 ['一个', '一体', '一门', '之间', '人工智能', '使用', '区别', '处理', '实现', '密切', '数学', '方向', '方法', '日常', '有效', '有着', '涉及', '特别', '理论', '研制', '研究', '科学', '联系', '自然语言', '计算机', '计算机科学', '计算机系统', '语言', '语言学', '软件系统', '这一', '进行', '通信', '重要', '领域']

vocabulary dic :

 {'区别': 6, '特别': 17, '一体': 1, '数学': 10, '方法': 12, '方向': 11, '计算机科学': 25, '研制': 19, '涉及': 16, '实现': 8, '日常': 13, '有着': 15, '语言学': 28, '这一': 30, '重要': 33, '人工智能': 4, '进行': 31, '理论': 18, '一门': 2, '自然语言': 23, '有效': 14, '通信': 32, '研究': 20, '联系': 22, '使用': 5, '科学': 21, '软件系统': 29, '计算机系统': 26, '领域': 34, '计算机': 24, '密切': 9, '之间': 3, '语言': 27, '一个': 0, '处理': 7}
vocabulary:


区别 6
特别 17
一体 1
数学 10
方法 12
方向 11
计算机科学 25
研制 19
涉及 16
实现 8
日常 13.....
```
(1) 停用词的配置：也可默认配置count_vec=CountVectorizer(stop_words=None) ，stop_words=None表示不去掉停用词；如果是英文的话，停用词不需要构建直接 count_vec=CountVectorizer(stop_words=’english’)则去掉英语停用词

(2)count_vec.fit_transform(data）的结果是如下的格式：

```python
print(count_vec.fit_transform(X_test))
 （0：data输入列表的元素索引（第几个文章（列表元素）），词典里词索引）  词频
  (0, 7)    2
  (0, 25)   2
  (0, 34)   2
  (0, 4)    1
  (0, 0)    1
  ......
print(count_vec.fit_transform(X_test).toarray())
[[1 1 1 1 1 0 0 2 1 0 1 1 1 0 1 0 0 0 1 0 1 1 0 3 1 2 0 0 1 0 0 1 1 1 2]
 [0 0 0 0 0 1 1 1 1 1 0 0 0 1 1 1 1 1 0 1 3 0 1 4 0 0 1 1 1 1 1 0 1 1 1]]

```
vocabulary_：字典类型，key为关键词，value是特征索引，样例如下：
com.furiousapps.haunt2: 57048
bale.yaowoo: 5025
asia.share.superayiconsumer: 4660
com.cooee.flakes: 38555
com.huahan.autopart: 67364
关键词集被存储为一个数组向量的形式，vocabulary_中的key是关键词，value就是该关键词在数组向量中的索引，使用get_feature_names()方法可以返回该数组向量。使用数组向量可验证上述关键词，如下：
```python
ipdb> count_vec.get_feature_names()[57048]
# 返回u'com.furiousapps.haunt2'
ipdb> count_vec.get_feature_names()[5025]
# 返回u'bale.yaowoo'
```
stop_words_：集合类型，官网的解释十分到位，如下：
Terms that were ignored because they either:
occurred in too many documents (max_df)
occurred in too few documents (min_df)
were cut off by feature selection (max_features).
This is only available if no vocabulary was given.
这个属性一般用来程序员自我检查停用词是否正确，在pickling的时候可以设置stop_words_为None是安全的。