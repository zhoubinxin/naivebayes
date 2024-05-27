import jieba
import multiprocessing as mp
from itertools import islice
from tqdm import tqdm

def load_stop_words():
    """
    加载停用词列表
    """
    stop_words = set()
    with open('./data/cnsmss/stopWord.txt', 'r', encoding='utf-8') as file:
        for line in file:
            stop_words.add(line.strip())
    return stop_words

def loadDataSet(stop_words, lines=5000):
    """
    读取中文数据集并进行预处理
    """
    postingList = []  # 存储文本
    classVec = []  # 存储标签
    with open('./data/cnsmss/80w.txt', 'r', encoding='utf-8') as file:
        dataSet = [line.strip().split('\t') for line in islice(file, lines)]
        for item in tqdm(dataSet, desc='加载数据集：'):
            # 检查数据格式是否正确，至少包含3个元素
            if len(item) >= 3:
                classVec.append(int(item[1]))  # 假设第2个元素是类别
                # 去除停用词
                words = jieba.lcut(item[2], cut_all=False)
                postingList.append([word for word in words if word not in stop_words])
            else:
                print(f"警告：数据行格式不正确，已跳过。原始行: '{item}'")
    return postingList, classVec

def preprocess_doc(args):
    """
    单个文档预处理函数，用于多进程调用
    """
    doc, stop_words = args
    return ' '.join(jieba.lcut(doc, cut_all=False) if isinstance(doc, str) else doc)  # 预处理文档并返回处理后的文本字符串

if __name__ == '__main__':
    # 测试nativeBayesCN
    stop_words = load_stop_words()
    listOposts, listClasses = loadDataSet(stop_words)
    print("Data loaded.")