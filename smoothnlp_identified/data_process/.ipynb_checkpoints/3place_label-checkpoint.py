import os
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import jieba
import jieba.posseg
import numpy as np

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
'''

#地点名词的扩充,跑的非常慢
def dict_extra():
    
    from item1_choose_blog.p_dict import pro_dict
     #需要另外加载一个词性标注模块
    '''
    string='上思县十万大山百鸟乐园景区'
    sentence_seged = jieba.posseg.cut(string.strip())
    outstr = ''
    for x in sentence_seged:
        outstr+="{}/{},".format(x.word,x.flag)
    print(outstr)
    '''

    pro=[]
    for key in pro_dict:
        pro.append(key)
    
    #先进行地点名词的扩充
    print(pro)
    for i in range(len(pro)):
        for string in pro_dict[pro[i]]:
            sentence_seged = jieba.posseg.cut(string)
            outstr = ''
            for x in sentence_seged:
                if x.flag in ['nr','ns','nt','nz','s','nl']:
                    pro_dict[pro[i]].append(x.word)

    return pro_dict

    
#作地点对照表
def place_table(pro_dict_extra={}):
    
    pro_key=['河北','山西','辽宁','黑龙江','江苏','浙江','安徽','福建','江西','山东','海南','河南','湖北','湖南','广东','四川','贵州','云南','陕西','甘肃','青海','内蒙古','广西','西藏','新疆','北京','天津','上海','重庆']
    ##其编号按照顺序从1开始编号
    
    try:
        from p_dict import pro_dict_extra  ##省份对应的景点数组
        #可以提前把字典打印好，否则跑起来非常慢
    
    if pro_dict_extra=={}:
        
        from item1_choose_blog.p_dict import pro_dict
        with open('data/dictionary_data/place_jieba_dic.xlsx','w',encoding='utf-8') as w:
            pro=1
            while(pro<31):
                w.write("%s\t%s\n" % (pro_dict[pro-1],pro))
                for p in pro_dict(pro_key[pro-1]):
                    w.write("%s\t%s\n" % (p,pro))
                    s = jieba.posseg.cut(p)
                    for x in s:
                        if x.flag in ['nr','ns','nt','nz','s','nl']:
                            w.write("%s\t%s\n" % (x.word,pro))
                pro=pro+1
    
    
    else:
        
        with open('data/dictionary_data/place_jieba_dic_extra.xslx','w',encoding='utf-8') as w:
            pro=1
            while(pro<31):
                w.write("%s\t%s\n" % (pro_dict_extra[pro-1],pro))
                for p in pro_dict_extra(pro_key[pro-1]):
                    w.write("%s\t%s\n" % (p,pro))
                    s = jieba.posseg.cut(p)
                    for x in s:
                        if x.flag in ['nr','ns','nt','nz','s','nl']:
                            w.write("%s\t%s\n" % (x.word,pro))
                pro=pro+1
'''
    
    
    
def place_table():
      from item1_choose_blog.p_dict import pro_dict
    import jieba.posseg #需要另外加载一个词性标注模块
    '''
    string='上思县十万大山百鸟乐园景区'
    sentence_seged = jieba.posseg.cut(string.strip())
    outstr = ''
    for x in sentence_seged:
        outstr+="{}/{},".format(x.word,x.flag)
    print(outstr)
    '''

    pro=[]
    for key in pro_dict:
        pro.append(key)

    print(pro)
    summ=[]
    plan=0
    with open('data/dictionary_data/place_jieba_dic.xlsx','w',encoding='utf-8') as w:
        for i in range(len(pro)):
            w.write('%s\t%s\n' % (pro[i],i+1))
            print(pro[i]+'开始')
            plan=1
            for string in pro_dict[pro[i]]:
                w.write('%s\t%s\n' % (string,i+1))
                plan=plan+1
                sentence_seged = jieba.posseg.cut(string)
                for x in sentence_seged:
                    if x.flag in ['nr','ns','nt','nz','s']:
                        w.write('%s\t%s\n' % (x.word,i+1))
                        plan=plan+1
            summ.append(plan)
            print(pro[i]+'已完成:'+plan)
            print('___________________')
            
#加入地点excel
def blog_corpus(input_file):
    in_ = os.path.join('data/merge_data/', input_file)####input.path设为'data/corpus'
    #out_ = os.path.join('data/original_data/', output_file)
    

    text = pd.read_excel(in_, 
            header=None,
            index_col=None,
            sep='\t'
            names=['id','web_id','blog','top','date','repost','coment','thumb','label','place'],
            dtype={'id':int,'name':str,'fans':int,'web_id':int,'blog':str,'time':str,'label':int})
                ##微博最后要留给评论的数据应该有：地点编号、微博id、时间
                ##微博需要分析的数值：粉丝数、评论数、转发数、点赞数、地点编号、时间
    place_label=pd.read_excel('data/dictionary_data/place_jieba_dic.csv', 
            header=None,
            names=['place','pro_id'],
            dtype={'place':string,'pro_id':int},
            sep='\t')
    
    model = gensim.models.Word2Vec.load(model_name)
    
    
    text = text.dropna().reset_index(drop=True)#Removing the missing values
    print("缺失值：",text.isnull().sum())
    text['province']=np.nan
        
    #text=read_f[['web_id','blog']]
    num=text.shape[0]
        
        
    file_name=input_file.split('.')
    train_xlsx=os.path.join('data/original_data/',file_name[0]+'_placed.xslx')
    
    
    pro_key=['河北','山西','辽宁','黑龙江','江苏','浙江','安徽','福建','江西','山东','海南','河南','湖北','湖南','广东','四川','贵州','云南','陕西','甘肃','青海','内蒙古','广西','西藏','新疆','北京','天津','上海','重庆']
      
    for i in range(num):
        #基于一个微博只有一个地点的假设
        place=text.loc[i,'place']
        place= place.split(' ')
        p_num=len(place)
        
        pro_id=0
        for p in place:
            if not place_label[place_label['place'] == p,'pro_id']:
                text.loc[i,'province']=place_label[place_label['place'] == p,'pro_id']
                pro_id=pro_id+1
        if pro_id<=0:
            s=[]
            for pro in pro_key:
                for p in place:
                    similar=model.wv.similarity(p,pro)
                    s.append(similar)
            s=np.array(s)
            s=s.reshape(30,-1)
            s=np.max(s,axis=1)
            #max=np.where(x==np.max(x))
            pro_id=np.argmax(s)+1
            text.loc[i,'province']=pro_id

                

    text.to_excel(tranin_xlsx,encoding='utf-8')

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
#jieba.load_userdict(file_name)
 
# 动态添加自定义词
#jieba.add_word(word, freq=None, tag=None)
 
# 动态删除自定义词
#jieba.del_word(word)
 
# 找到合适词频
#jieba.suggest_freq(segment, tune=True)


#transfor('blog.txt')
#blog_corpus('blog_100tho.csv','blog_id_100tho.csv','blog_corpus_100tho.txt')


#x,num=blog_corpus('blog.csv','blog_corpus_14')
#print(num)



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
    place_table()
    
    corpus,num=blog_corpus('merge1.xslx','merge1_processed.xslx')
    print(num)
#print(sentence_depart('我爱北京天安门啊啊啊啊啊啊啊[爱心]'))