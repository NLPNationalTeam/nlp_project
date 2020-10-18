#数据最开始要分成两类，训练集和测试集
import re
import jieba.analyse
import jieba.posseg #需要另外加载一个词性标注模块
import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import h5py



import re
import os




# 对句子去除停用词,返回列表
def movestopwords(sentence):
    stopwords = [line.strip() for line in open('data/stopwords_hagongda.txt', 'r', encoding='utf-8').readlines()] # 这里加载停用词的路径
    santi_words =[x for x in sentence if x !=None and x not in stopwords]
    
    return santi_words

# 分词,除去停用词
def sentence_depart(sentence):##################这里注意要识别表情！！！识别话题！！！这里的表情没有去掉[]
    sentence=sentence.strip()
    sentence=sentence.strip('\n'+'...展开全文c')
    
    topic=re.finditer('#[\w]*#',sentence)#【[\w]*】的话题分类就太细了
    topics=''   
    for t in topic:
        a=t.group()
        a=a.strip('#')
        topics=topics+a+' '
    topics=topics.strip()
    
    sentence=re.sub('\[[\w]*\]', '', sentence, count=0, flags=0)
    sentence=jieba.posseg.cut(sentence)
    #jieba.enable_parallel()   ##jieba并行处理，但是不支持windows系统
    outstr=[]
    place=[]
    for i in sentence:
        if i.word != '\t'and i.flag in ['n','nr','ns','nt','nz','nl','s','f','v','vd','vn','vg','vl','a','ad','an','ag','al','d','b']:
            outstr.append(i.word)
            if i.flag in['nr','ns','nt','nz','nl','s']:
                place.append(i.word)
    outstr=movestopwords(outstr)
    place=movestopwords(place)
    outstr=' '.join(outstr)
    place=' '.join(place)
    
    
    
    result={'blog':outstr,'topic':topics,'place':place}    
    return result


#处理标记的数据，用于训练分类模型，加入地点 分好词 存成txt csv
def blog_corpus(input_file):
    in_ = os.path.join('data/merge_data/', input_file)
    text = pd.read_excel(in_, 
            header=0,
            index_col=None,
            sep='\t',
            names=['id','name','fans','web_id','blog','time','repost','coment','thumb','label'],
            dtype={'id':int,'name':str,'fans':int,'web_id':int,'blog':str,'time':str,'label':int})
                ##微博最后要留给评论的数据应该有：地点编号、微博id、时间
                ##微博需要分析的数值：粉丝数、评论数、转发数、点赞数、地点编号、时间
                ##注意检查一下header这个参数
    print(text.head())
    text = text.dropna().reset_index(drop=True)#Removing the missing values
    print("缺失值：",text.isnull().sum())
        

    num=text.shape[0]
    
    text['topic']=None
    text['place']=None
    l=['id','name','fans','web_id','blog','topic','time','repost','coment','thumb','label','place']
    text=text[l]
    print('最终表的列名：',text.columns.values)
        
    file_name=input_file.split('.')
    train_xlsx=os.path.join('data/original_data/',file_name[0]+'_processed.xlsx')
    txt=os.path.join('data/original_data/',file_name[0]+'_processed.txt')
    
    
    with open (txt,'w',encoding='utf-8') as t:
        for i in range(num):
            #分词
            answer=sentence_depart(text.loc[i,'blog'])            
            blog=answer['blog']
            
            text.loc[i,'blog']=answer['blog']
            text.loc[i,'topic']=answer['topic']
            text.loc[i,'place']=answer['place']
            #top=sentence_depart(text.loc[i,'top']) ##topic先试试直接模糊对比，再试试用余弦计算两者相似度
            
            t.write('%s\n' % blog)
            
    
    text.to_excel(train_xlsx,encoding='utf-8',index=False)
            
    
    return num

#处理全体语料：评论和微博，用于训练词向量
def unlabeld_blog_corpus(input_file):
    in_ = os.path.join('data/big_corpus/', input_file)
    text = pd.read_excel(in_, 
            header=0,
            index_col=None,
            sep='\t',
            names=['id','name','fans','web_id','blog','time','repost','coment','thumb'],
            dtype={'id':int,'name':str,'fans':int,'web_id':int,'blog':str,'time':str})
                ##微博最后要留给评论的数据应该有：地点编号、微博id、时间
                ##微博需要分析的数值：粉丝数、评论数、转发数、点赞数、地点编号、时间
    
    print(text.head())
    text = text.dropna().reset_index(drop=True)#Removing the missing values
    print("缺失值：",text.isnull().sum())
    text['place']=None
    text['topic']=None
    
    col=['id','name','fans','web_id','blog','topic','time','repost','coment','thumb','place']
    text=text[col]        
    #text=read_f[['web_id','blog']]
    num=text.shape[0]
    
    
        
    file_name=input_file.split('.')
    train_xlsx=os.path.join('data/big_corpus/',file_name[0]+'_processed.xlsx')
    txt=os.path.join('data/big_corpus/',file_name[0]+'_processed.txt')
    tt='data/big_corpus/blog_topic.txt'
    
    
    with open (txt,'w',encoding='utf-8') as t,open (tt,'w',encoding='utf-8') as to:
        for i in range(num):
            #top=sentence_depart(text.loc[i,'top']) ##topic先试试直接模糊对比，再试试用余弦计算两者相似度
            
            #分词
            answer=sentence_depart(text.loc[i,'blog'])            
            blog=answer['blog']
            topic=answer['topic']
            topic=topic.split(' ')
            for i in topic:
                to.write('%s\n'% i)
            
            text.loc[i,'blog']=answer['blog']
            text.loc[i,'topic']=answer['topic']
            text.loc[i,'place']=answer['place']
            #top=sentence_depart(text.loc[i,'top']) ##topic先试试直接模糊对比，再试试用余弦计算两者相似度
            
            t.write('%s\n' % blog)
            #writer.write("%s\n" % blog)

    
    print('新的表已生成：/n',text.head())
    text.to_excel(train_xlsx,encoding='utf-8',index=False)
    
    
    return num


if __name__ == "__main__":
    
    num=unlabeld_blog_corpus('merge_blog.xlsx')
    print(num)
