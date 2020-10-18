import sys
import numpy as np
import gensim
import pandas as pd
from time import time  # To time our operations
import logging  # Setting up the loggings to monitor gensim
import multiprocessing

from gensim.models.doc2vec import Doc2Vec,LabeledSentence,TaggedDocument,TaggedLineDocument
#from sklearn.cross_validation import train_test_split

TaggedDocument = gensim.models.doc2vec.TaggedDocument
TaggedLineDocument=gensim.models.doc2vec.TaggedLineDocument

##读取并预处理数据
def get_dataset():
    #读取数据
    df = pd.read_csv('data/blog_corpus.csv',header=None,names=['blog','topic'])
    blogs=list(df['blog'])
    '''
    #使用1表示正面情感，0为负面
    y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
    #将数据分割为训练与测试集
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)
    '''

    #Gensim的Doc2Vec应用于训练要求每一篇文章/句子有一个唯一标识的label.
    #我们使用Gensim自带的LabeledSentence方法. 标识的格式为"TRAIN_i"和"TEST_i"，其中i为序号
    def labelizeReviews(reviews):
        labelized = []
        for i,v in enumerate(reviews):
            #label = '%s_%s'%(label_type,i)
            labelized.append(TaggedDocument(v, tags=i))
        return labelized

    x_train = labelizeReviews(blogs)

    return x_train #x_test,unsup_reviews,y_train, y_test

def dataset():
    df = open('data/blog_notopic.txt','r',encoding='utf-8')
    blogs=TaggedLineDocument(df)
    
    #计算x_train数据量
    '''count=-1
    for count, line in enumerate(open('data/blog_notopic.txt', 'r',encoding='utf-8')):
        pass
    count += 1'''
    return blogs
    

##读取向量
def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

##对数据进行训练
def train(x_train,size = 300,epoch_num=10):
    
    logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
    cores = multiprocessing.cpu_count() # Count the number of cores in a computer
    
    #实例DM和DBOW模型
    model_dm = gensim.models.Doc2Vec(alpha=0.003, min_alpha=0.0007,min_count=1, window=5, size=size, sample=1e-3, negative=10,dm=1, workers=cores-1)
    model_dbow = gensim.models.Doc2Vec(alpha=0.003, min_alpha=0.0007,min_count=1, window=5, size=size, sample=1e-3, negative=10, dm=0, workers=cores-1)

    #使用所有的数据建立词典
    model_dm.build_vocab(x_train)
    model_dbow.build_vocab(x_train)

    #进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    '''x_train=list(x_train)
    for epoch in range(epoch_num):
        perm = np.random.permutation(len(x_train))
        model_dm.train(x_train[perm])
        model_dbow.train(x_train[perm])'''
    '''
    #训练测试数据集
    x_test = np.array(x_test)
    for epoch in range(epoch_num):
        perm = np.random.permutation(x_test.shape[0])
        model_dm.train(x_test[perm])
        model_dbow.train(x_test[perm])
    '''
    
    model_dm.save('doc2vec_dm')
    model_dbow.save('doc2vec_dbow')

    return model_dm,model_dbow

##将训练完成的数据转换为vectors
def get_vectors(model_dm,model_dbow,size):

    #获取训练数据集的文档向量
    train_vecs_dm = getVecs(model_dm, x_train, size)
    train_vecs_dbow = getVecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    '''
    #获取测试数据集的文档向量
    test_vecs_dm = getVecs(model_dm, x_test, size)
    test_vecs_dbow = getVecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))    
    '''

    return train_vecs


if __name__=='__main__':
    x_train=dataset()
    model_dm,model_dbow=train(x_train,size = 300,epoch_num=10)
    #model_dm=gensim.models.doc2vec.Doc2Vec('doc2vec_dm')
    #model_dbow=gensim.models.doc2vec.Doc2Vec('doc2vec_dbow')
    y=get_vectors(model_dm,model_dbow,size = 300)
    print(y.shape)