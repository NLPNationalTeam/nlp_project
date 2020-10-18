###Gensim 本身只是要求能够迭代的有序句子列表，因此在工程实践中我们可以使用自定义的生成器，只在内存中保存单条语句。
###把英文改成中文，单词的词向量变成句子向量，把
import re  # For preprocessing
import os
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import spacy  # For preprocessing
import h5py
import numpy as np
import gensim
from gensim.models import word2vec

def get_trainset(x_train):
    from gensim import corpora
    
    x=[]
    TaggededDocument = gensim.models.doc2vec.TaggedDocument # 输入输出内容都为 词袋 + tag列表， 作用是记录每一篇博客的大致内容，并给该博客编号
    for i,words in enumerate(x_train):
        x.append(TaggededDocument(words, tags=[i])) 

    
    # 赋给语料库中每个词(不重复的词)一个整数id
    dictionary = corpora.Dictionary(x)
    print(dictionary)
    # 通过下面的方法可以看到语料库中每个词对应的id
    #print(dictionary.token2id)
    new_corpus = [dictionary.doc2bow(text) for text in x_train]
    return new_corpus,dictionary
    
#实现shuffle
def w_to_vec(x_train,epoch_num,model_name):
    
    x,dictionary=get_trainset(x_train)
    #x=np.array(x)
    
    import multiprocessing
    from gensim.models import Word2Vec
    cores = multiprocessing.cpu_count() # Count the number of cores in a computer

    model = word2vec.Word2Vec(min_count=5,
                        window=2,
                        size=300,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=cores-1)
    
    #使用所有的数据建立词典
    t = time()
    model.build_vocab(x, progress_per=10000)#可以试试一个的效果：blog，blog+topic,topic
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    #进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    for epoch in range(epoch_num):
        t = time()
        perm = np.random.permutation(x.shape[0])
        model.train(x[perm])
        print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)),epoch)
    
    model.save(model_name) 



def word2vec_train(corpus_file,size_=300):
    import logging  # Setting up the loggings to monitor gensim
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
    
    '''
    print(df.head())
    print('文件空的行数：'df.isnull().sum())
    df = df.dropna().reset_index(drop=True)
    print('整理后，文件空的行数：',df.isnull().sum())
    blogs=list(df['blog'])
    topic=list(df['topic'])
    '''    
    
    blogs = word2vec.Text8Corpus(os.path.join('data/big_corpus',corpus_file)) 

    import multiprocessing
    from gensim.models import Word2Vec
    cores = multiprocessing.cpu_count() # Count the number of cores in a computer

    '''
    The parameters:
    min_count = int - Ignores all words with total absolute frequency lower than this - (2, 100)
    window = int - The maximum distance between the current and predicted word within a sentence. E.g. window words on the left and window words on the left of our target - (2, 10)
    size = int - Dimensionality of the feature vectors. - (50, 300)
    sample = float - The threshold for configuring which higher-frequency words are randomly downsampled. Highly influencial. - (0, 1e-5)
    alpha = float - The initial learning rate - (0.01, 0.05)
    min_alpha = float - Learning rate will linearly drop to min_alpha as training progresses. To set it: alpha - (min_alpha * epochs) ~ 0.00
    negative = int - If > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drown. If set to 0, no negative sampling is used. - (5, 20)
    workers = int - Use these many worker threads to train the model (=faster training with multicore machines)
    '''

    model = word2vec.Word2Vec(min_count=5,
                        window=2,
                        size=size_,
                        sample=6e-5, 
                        alpha=0.03, 
                        min_alpha=0.0007, 
                        negative=20,
                        workers=cores-1)

    t = time()
    model.build_vocab(blogs, progress_per=10000)#可以试试一个的效果：blog，blog+topic,topic
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()
    model.train(blogs, total_examples=model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))


    #from gensim.keyedvectors import KeyedVectors
    #save
    filename=corpus_file.split('.')
    model.save('word2v_'+str(size_)) 
    
    return model
    
    #model.accuracy('./datasets/questions-words.txt')
    #Word2Vec培训是一项无监督的任务，没有一种很好的方法可以客观地评估结果。评估取决于最终应用程序。


def sen_matrix(corpus_csv,model=None):
    if not model:
        filename=corpus_csv.split('.')
        model_name='word2v_'+filename[0]
        model = gensim.models.Word2Vec.load(model_name) 
    try:
        f=pd.read_csv(os.path.join('data',corpus_csv),names=['web_id','blog','topic','label'],header=None)
    except:
        f=pd.read_csv(os.path.join('data',corpus_csv),names=['web_id','blog','topic'],header=None)
        
    df=pd.read_csv(os.path.join('data/original data',corpus_csv),error_bad_lines=False)###逗号分隔有问题
    
    '''print('文件空的行数：'f.isnull().sum())
    f = f.dropna().reset_index(drop=True)
    print('整理后，文件空的行数：',f.isnull().sum())'''
    
    blog=list(f['blog'])
    blog_vec=[]
    sen_len=[]
    nan_idx=[]
    for i,words in enumerate(blog):
        num=1
        if type(words)==str:
            words = words.strip().split()
            w_vec=[]
            for w in words:
                w=w.strip()
                try:
                    w_vec.append(model.wv[w])
                except:
                    continue
            sen_len.append(len(w_vec))
            blog_vec.append(w_vec)
        else:
            nan_idx.append(i)
            if num:
                print(df.iloc[i,4])
                num=0
                            
    print('句子长度:',sen_len)
    print('句子矩阵',blog_vec)
    
    
    


if __name__ == '__main__':
    
    for i in range(100,450,50):
        model=word2vec_train('blog_processed.xlsx',i)
    #sen_matrix('blog_11w.csv')


