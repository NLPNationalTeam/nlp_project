# 数据最开始要分成两类，训练集和测试集
import re
import jieba.analyse
import jieba.posseg  # 需要另外加载一个词性标注模块
import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import h5py
import re
import os

# 把数据随机分成训练集和测试集，把训练集随机分成训练集和验证集

def split_csv(infile, trainfile, valtestfile, seed=999, ratio=0.2):
    '''
    infile:待分割的csv文件

    trainfile:分割出的训练cs文件

    valtestfile：分割出的测试或验证csv文件

    seed:随机种子，保证每次的随机分割随机性一致

    ratio:测试（验证）集占数据的比例

    '''

    df = pd.read_csv(infile)
    df["text"] = df.text.str.replace("\n", " ")
    idxs = np.arange(df.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idxs)
    val_size = int(len(idxs) * ratio)
    df.iloc[idxs[:val_size], :].to_csv(valtestfile, index=False)


# 对句子去除停用词,返回列表
def movestopwords(sentence):
    stopwords = [line.strip() for line in open('data/stopwords_hagongda.txt', 'r', encoding='utf-8').readlines()]  # 这里加载停用词的路径
    santi_words = [x for x in sentence if x is not None and x not in stopwords]
    # s=''.join(santi_words)

    return santi_words

# 分词,除去停用词


def sentence_depart(sentence):  # 这里注意要识别表情！！！识别话题！！！这里的表情没有去掉[]
    sentence = sentence.strip()
    sentence = sentence.strip('\n' + '...展开全文c')

    '''
    emoji=re.finditer(r'\[[\w]*\]',sentence)
    sentence_emoji=[]
    for emo in emoji:
        a=emo.group()#
        a=re.sub(r'[\[\]]', '', a, count=0, flags=0)
        sentence_emoji.append(a+'/emoj')#
        #sentence.replace(emo.group(),'')
    '''

    # #[xxx]
    topic = re.finditer(r'#[\w]*#', sentence)  # 【[\w]*】的话题分类就太细了
    topics = ''
    for t in topic:
        a = t.group()
        a = a.strip('#')
        topics = topics + a + ' '
    topics = topics.strip()

    sentence = re.sub(r'\[[\w]*\]', '', sentence, count=0, flags=0)
    # sentence= movestopwords(sentence)
    sentence = jieba.posseg.cut(sentence)
    # jieba.enable_parallel()   ##jieba并行处理，但是不支持windows系统
    outstr = []
    place = []
    for i in sentence:
        if i.word != '\t' and i.flag in ['n', 'nr', 'ns', 'nt', 'nz', 'nl', 's', 'f', 'v', 'vd', 'vn', 'vg', 'vl', 'a', 'ad', 'an', 'ag', 'al', 'd', 'b']:
            outstr.append(i.word)
            if i.flag in ['nr', 'ns', 'nt', 'nz', 'nl', 's']:
                place.append(i.word)
    outstr = movestopwords(outstr)
    place = movestopwords(place)
    outstr = ' '.join(outstr)
    place = ' '.join(place)

    '''
    outstr=''
    for i in sentence:
        if i.word != '\t':
            outstr += i.word+'/'+i.flag
            outstr += " "
    outstr=outstr.strip()
    '''

    '''
    for emo in sentence_emoji:
        if emo == sentence_emoji[0]:
            outstr=outstr+''+emo
        else:
            outstr=outstr+' '+emo
    '''
    #outstr =outstr.encode(utf-8)

    result = {'blog': outstr, 'topic': topics, 'place': place}
    return result


# 加入地点 分好词 存成txt csv
def blog_corpus(input_file):
    in_ = os.path.join('data/merge_data/', input_file)  # input.path设为'data/corpus'
    #out_ = os.path.join('data/original_data/', output_file)
    text = pd.read_excel(in_,
                         header=0,
                         index_col=None,
                         sep='\t',
                         names=['id', 'name', 'fans', 'web_id', 'blog', 'time', 'repost', 'coment', 'thumb', 'label'],
                         dtype={'id': int, 'name': str, 'fans': int, 'web_id': int, 'blog': str, 'time': str, 'label': int})
    # 微博最后要留给评论的数据应该有：地点编号、微博id、时间
    # 微博需要分析的数值：粉丝数、评论数、转发数、点赞数、地点编号、时间
    # 注意检查一下header这个参数
    print(text.head())
    text = text.dropna().reset_index(drop=True)  # Removing the missing values
    print("缺失值：", text.isnull().sum())

    # text=read_f[['web_id','blog']]
    num = text.shape[0]

    text['topic'] = None
    text['place'] = None
    l = ['id', 'name', 'fans', 'web_id', 'blog', 'topic', 'time', 'repost', 'coment', 'thumb', 'label', 'place']
    text = text[l]
    print('最终表的列名：', text.columns.values)

    file_name = input_file.split('.')
    train_xlsx = os.path.join('data/original_data/', file_name[0] + '_processed.xlsx')
    txt = os.path.join('data/original_data/', file_name[0] + '_processed.txt')

    with open(txt, 'w', encoding='utf-8') as t:
        for i in range(num):
            # 分词
            answer = sentence_depart(text.loc[i, 'blog'])
            blog = answer['blog']

            text.loc[i, 'blog'] = answer['blog']
            text.loc[i, 'topic'] = answer['topic']
            text.loc[i, 'place'] = answer['place']
            # top=sentence_depart(text.loc[i,'top']) ##topic先试试直接模糊对比，再试试用余弦计算两者相似度

            t.write('%s\n' % blog)
            #writer.write("%s\n" % blog)

    text.to_excel(train_xlsx, encoding='utf-8', index=False)
    '''
            #基于一个微博只有一个地点的假设
            place=re.findall(r'[\w]*/nr|[\w]*/ns|[\w]*/nt|[\w]*/nz|[\w]*/s',blog+' '+top)
            pla=[]
            for i,p in enumerate(place):
                new = re.compile('/[a-zA-Z]+')
                p = new.sub('',p)#p.group()
                place[i]=p
            place=' '.join(place)
    '''

    '''
            #按照固定词性留下词语
            final=re.findall('[\w]*/n|[\w]*/ns|[\w]*/nt|[\w]*/nz|[\w]*/ng|[\w]*/l|[\w]*/i|[\w]*/t|[\w]*/tg|[\w]*/s|[\w]*/f|[\w]*/v|[\w]*/vd|[\w]*/vn|[\w]*/vg|[\w]*/a|[\w]*/ad|[\w]*/an|[\w]*/ag|[\w]*/d|[\w]*/b|[\w]*/r|[\w]*/m|[\w]*/q|[\w]*/p',
                            blog,
                            re.S)###跳过转行符
            if not blog:
                print(i)
                nan_list.append(i)
            c=[]
            for i in final:
                i=re.search(r'[\w]*',i).group()
                c.append(i)
            blog=' '.join(c)
            blog=blog.strip()
            corpus.append(blog)
    '''

    return num

    '''
            words=jieba.posseg.cut(text.loc[i,'blog'])
            result = []
            for i in seg:
                if i.flag=='x':
                    pass
                else:
                    result.append('/'.join(i))
            result=' '.join(result)
            text.at[i,'blog']=result
    '''

# 自定义词典中：词语、词频（可省略）、词性（可省略），用空格隔开，顺序不可颠倒
# 李小福 2 nr
# file_name为文件类对象或自定义词典的路径
# jieba.load_userdict(file_name)

# 动态添加自定义词
#jieba.add_word(word, freq=None, tag=None)

# 动态删除自定义词
# jieba.del_word(word)

# 找到合适词频
#jieba.suggest_freq(segment, tune=True)


# transfor('blog.txt')
# blog_corpus('blog_100tho.csv','blog_id_100tho.csv','blog_corpus_100tho.txt')


# x,num=blog_corpus('blog.csv','blog_corpus_14')
# print(num)
# 加入地点 分好词 存成txt csv


def unlabeld_blog_corpus(input_file):
    in_ = os.path.join('data/big_corpus/', input_file)  # input.path设为'data/corpus'
    chunker = pd.read_excel(in_,
                         header=0,
                         index_col=None,
                         # sep='\t',
                         names=['id', 'name', 'fans', 'web_id', 'blog', 'time', 'repost', 'coment', 'thumb'],
                         dtype={'id': int, 'name': str, 'fans': int, 'web_id': int, 'blog': str, 'time': str},
                         chunksize=10000)

    for text in chunker:

        # 微博最后要留给评论的数据应该有：地点编号、微博id、时间
        # 微博需要分析的数值：粉丝数、评论数、转发数、点赞数、地点编号、时间

        print(text.head())
        text = text.dropna().reset_index(drop=True)  # Removing the missing values
        print("缺失值：", text.isnull().sum())
        text['place'] = None
        text['topic'] = None

        col = ['id', 'name', 'fans', 'web_id', 'blog', 'topic', 'time', 'repost', 'coment', 'thumb', 'place']
        text = text[col]
        # text=read_f[['web_id','blog']]
        num = text.shape[0]

        file_name = input_file.split('.')
        train_xlsx = os.path.join('data/big_corpus/', file_name[0] + '_processed.xlsx')
        txt = os.path.join('data/big_corpus/', file_name[0] + '_processed.txt')
        tt = 'data/big_corpus/blog_topic.txt'

        with open(txt, 'w', encoding='utf-8') as t, open(tt, 'w', encoding='utf-8') as to:
            for i in range(num):
                # top=sentence_depart(text.loc[i,'top']) ##topic先试试直接模糊对比，再试试用余弦计算两者相似度

                # 分词
                answer = sentence_depart(text.loc[i, 'blog'])
                blog = answer['blog']
                topic = answer['topic']
                topic = topic.split(' ')
                for i in topic:
                    to.write('%s\n' % i)

                text.loc[i, 'blog'] = answer['blog']
                text.loc[i, 'topic'] = answer['topic']
                text.loc[i, 'place'] = answer['place']
                # top=sentence_depart(text.loc[i,'top']) ##topic先试试直接模糊对比，再试试用余弦计算两者相似度

                t.write('%s\n' % blog)
                #writer.write("%s\n" % blog)

        print('新的表已生成：/n', text.head())
        text.to_excel(train_xlsx, encoding='utf-8', index=False)

        return num


if __name__ == "__main__":

    num = unlabeld_blog_corpus('merge_blog.xlsx')
    # corpus,num=unlabeld_blog_corpus('merge_blog.xlsx','blog_processed.xlsx')
    print(num)
# print(sentence_depart('我爱北京天安门啊啊啊啊啊啊啊[爱心]'))
