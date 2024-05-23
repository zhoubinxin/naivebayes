# naiveBayes算法
from tqdm import tqdm
import jieba
import multiprocessing as mp
from itertools import islice


def load_stop_words():
    """
    加载停用词
    :return:
    """
    stop_words = set()
    with open('./data/cnsmss/stopWord.txt', 'r', encoding='utf-8') as file:
        for line in file:
            stop_words.add(line.strip())
    return stop_words


def loadDataSet(stop_words, lines=5000):
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
            words = jieba.lcut(item[2], cut_all=False)
            postingList.append(words)
        except IndexError:
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


def createVocabList(dataSet):
    """
    提取数据集中的单词列表

    :param dataSet:
    :return:
    """
    # 分割数据集以便多进程处理
    num_processes = mp.cpu_count()  # 获取CPU核心数量
    chunk_size = len(dataSet) // num_processes

    # 将数据集分割成多个块，每个块大小为 chunk_size
    chunks = [dataSet[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
    # 如果数据集不能被核心数量整除，则将剩余数据添加到最后一个块中
    if len(dataSet) % num_processes != 0:
        chunks.append(dataSet[num_processes * chunk_size:])

    with mp.Pool(processes=num_processes) as pool:
        # 使用 imap_unordered 并行处理每个数据块
        results = list(
            tqdm(pool.imap_unordered(vocab_process, chunks), total=len(chunks), desc='创建词汇表：')
        )

    # 将所有结果合并
    vocabSet = set().union(*results)
    return list(vocabSet)


def vocab_process(chunk):
    vocabSet = set()
    for document in chunk:
        vocabSet = vocabSet | set(document)
    return vocabSet


def main():
    stop_words = load_stop_words()
    print(stop_words)


if __name__ == '__main__':
    main()
