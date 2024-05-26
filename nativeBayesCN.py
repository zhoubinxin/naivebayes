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
    加载并预处理中文数据集
    """
    postingList = []
    classVec = []
    with open('./data/cnsmss/80w.txt', 'r', encoding='utf-8') as file:
        dataSet = [line.strip().split('\t') for line in islice(file, lines)]

    for item in tqdm(dataSet, desc='加载数据集：'):
        # 0：非垃圾短信；1：垃圾短信
        classVec.append(int(item[1]))

        # 将每条短信拆分为单词列表
        try:
            words = jieba.lcut(item[2], cut_all=False)
            postingList.append(words)
        except IndexError:
            postingList.append('')
            # 空文本
            pass

    return postingList, classVec

def preprocess_doc(args):
    """
    单个文档预处理函数，用于多进程调用
    """
    doc, stop_words = args
    return ' '.join(jieba.lcut(doc, cut_all=False) if isinstance(doc, str) else doc)  # 预处理文档并返回处理后的文本字符串

if __name__ == '__main__':
    # 这里仅用于演示或测试nativeBayesCN模块的功能，实际应用中不需要这部分代码
    stop_words = load_stop_words()
    listOposts, listClasses = loadDataSet(stop_words)
    print("Data loaded.")