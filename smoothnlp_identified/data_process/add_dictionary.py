import os
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import jieba
import jieba.posseg
import numpy as np

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)



#添加地点名词
def create_place_dic():
    place_corpus=open('data/dictionary_data/place_merge.txt','r',encoding='utf-8')
    line=place_corpus.read()
    line.replace(' ','')
    places=re.finditer(u'[\u4e00-\u9fa5]+',line)######读取中文
    with open('data/dictionary_data/place_jieba_dic.txt','w',encoding='utf-8') as w: 
        for place in places:
            w.write(place.group()+' ns'+'\n')

def add_place_jieba():
    
    corpus = open('data/dictionary_data/place_jieba_dic.txt', 'r', encoding='utf-8')
    pro_key=['河北','山西','辽宁','黑龙江','江苏','浙江','安徽','福建','江西','山东','海南','河南','湖北','湖南','广东','四川','贵州','云南','陕西','甘肃','青海','内蒙古','广西','西藏','新疆','北京','天津','上海','重庆']
            
    
    jieba.load_userdict(corpus)
    print('jieba loads!')


    
#添加网络名词
def add_dictionary(dictionary=None,num=0,top=0,min_=0):############smoothnlp版本，要根据具体的文件格式来改
    
    #已存在73w的语料库
    if dictionary==None:
        output_corpus = 'data/dictionary.txt'
        dic_file='data/dictionary_data/web_dictionary'+num+'.txt'
        
        corpus = open(output_corpus, 'r', encoding='utf-8')
    
        from smoothnlp_identified.algorithm.phrase import extract_phrase
    
        if min_>0:
            dic=extract_phrase(corpus,top,min_freq=min_)#################也可以从数据库直接读取文件，定一下min_freq   
        else:
            dic=extract_phrase(corpus,top)
        #extract_phrase(corpus,top_k,chunk_size,min_n,max_n,min_freq)
        with open(dic_file, 'w', encoding='utf-8') as writer:
            for word in dic:
                print(word)
                writer.write(word+' n'+'\n')
    else:
        dic_file=dictionary
        
    jieba.load_userdict(dic_file)
    print('jieba loads!')


if __name__ == "__main__":
    
    print('jieba正在添加景点名词')
    add_place_jieba()
    print('正在添加网络用词')
    add_dictionary('data/dictionary_data/web_dictionary.txt')