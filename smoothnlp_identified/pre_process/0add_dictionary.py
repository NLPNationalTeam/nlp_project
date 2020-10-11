import os
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import jieba
import jieba.posseg
import numpy as np
import re

from smoothnlp_identified.smoothnlp_identified.algorithm.phrase.phrase_extraction import extract_phrase
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt='%H:%M:%S', level=logging.INFO)


# 添加地点名词
def create_place_dic():
    place_corpus = open('data/dictionary_data/place_merge.txt', 'r', encoding='utf-8')
    line = place_corpus.read()
    line.replace(' ', '')
    places = re.finditer(u'[\u4e00-\u9fa5]+', line)  # 读取中文
    with open('data/dictionary_data/place_jieba_dic.txt', 'w', encoding='utf-8') as w:
        for place in places:
            w.write(place.group() + ' ns' + '\n')


def add_place_jieba():

    corpus = open('data/dictionary_data/place_jieba_dic.txt', 'r', encoding='utf-8')
    pro_key = ['河北', '山西', '辽宁', '黑龙江', '江苏', '浙江', '安徽', '福建', '江西', '山东', '海南', '河南', '湖北', '湖南', '广东', '四川', '贵州', '云南', '陕西', '甘肃', '青海', '内蒙古', '广西', '西藏', '新疆', '北京', '天津', '上海', '重庆']
    jieba.load_userdict(corpus)
    #print('jieba loads %s!'% dic_name)
    print('jieba loads!')


# 读取文件为list，使得其可以区块提取
def ReadTxtName(rootdir):
    lines = []
    with open(rootdir, 'r', encoding='utf-8') as file_to_read:
        while True:
            line = file_to_read.readline()
            if not line:
                break
            line = line.strip('\n')
            lines.append(line)
    return lines

# 添加名词
def add_dictionary(num, top, min_=0):  # smoothnlp版本，要根据具体的文件格式来改

    # 已存在73w的语料库
    output_corpus = 'data/dictionary.txt'
    dic_file = 'data/dictionary_data/web_dictionary' + 'my'+ '.txt'

    corpus = ReadTxtName(output_corpus)

    dic = []

    for i in range(100):
        num = int(len(corpus)/100)

        if min_ > 0:
            t_dic = extract_phrase(corpus[ i*num: i*(num+1)], top, min_freq=min_)  # 也可以从数据库直接读取文件，定一下min_freq
        else:
            t_dic = extract_phrase(corpus[ i*num: i*(num+1)], top)
        print('提取第i个')

        dic.extend(t_dic)


    # extract_phrase(corpus,top_k,chunk_size,min_n,max_n,min_freq)
    with open(dic_file, 'w', encoding='utf-8') as writer:
        for word in dic:
            print(word)
            writer.write(word + ' n' + '\n')
    jieba.load_userdict(dic_file)
    #print('jieba loads %s!'% dic_name)
    print('jieba loads!')


'''
    多个文件输入作为语料
    for root,dirs,files in os.walk('data/original_data'):#####################导入语料的文件夹，可以考虑在主函数用pool多线程处理
        data_set=files #列表
        with open(output_corpus, 'w') as writer:
            for file_name in data_set:
                input_corpus=os.path.join('data/original_data', file_name)
            with open(input_corpus, 'r') as f:
                for line in f:
                    items = line.strip().split(',')
                    comment=item[-2]#评论数据在倒数第二列
                    if comment[-1] not in ['。','！','？','~']:
                        comment=comment+'。'
                    writer.write(comment)##########注意当前的光标位置
'''

if __name__ == "__main__":

    '''
    j=0
    print(np.linspace(0.0005,0.005,num=20))
    for i in np.linspace(0.0005,0.005,num=20):
        j=j+1
        num=str(j)
        add_dictionary(num,i)#73w数据平均每句有20个词的估算方式 #0.0005 0.005

    print('OK')
    '''
    print('正在添加网络用词')
    add_dictionary('0', 0.995, 0.00000001)
    print('jieba正在添加景点名词')
    add_place_jieba()
