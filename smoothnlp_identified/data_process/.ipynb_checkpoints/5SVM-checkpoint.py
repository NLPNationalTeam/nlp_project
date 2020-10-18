from sklearn import svm
import numpy as np

'''
https://blog.csdn.net/u012679707/article/details/80511968
用np读取txt文件
  常用的参数有：

        fname: 文件路径，例 path='F:/Python_Project/SVM/data/Iris.data'

   dtype:样本的数据类型 例dtype=float


        delimiter：分隔符。例 delimiter=','

        converters：将数据列与转换函数进行映射的字典。例 converters={4:Iris_label}含义是将第5列的数据对应转换函数进行转换。

        usecols：选取数据的列。
'''


x=np.load('word2vec/svm_train_x')
y=np.load('word2vec/svm_train_y')
test_x=np.load('word2vec/svm_text_x')

train_data,test_data,train_label,test_label =sklearn.model_selection.train_test_split(x,y, random_state=1, train_size=0.7,test_size=0.3)

'''
      1. split(数据，分割位置，轴=1（水平分割） or 0（垂直分割）)。
　　2.  sklearn.model_selection.train_test_split随机划分训练集与测试集。train_test_split(train_data,train_label,test_size=数字, random_state=0)

　　参数解释：

　    　train_data：所要划分的样本特征集

　    　train_label：所要划分的样本类别

　    　test_size：样本占比，如果是整数的话就是样本的数量.(注意：)

                   --  test_size:测试样本占比。 默认情况下，该值设置为0.25。 默认值将在版本0.21中更改。 只有train_size没有指定时， 

                        它将保持0.25，否则它将补充指定的train_size，例如train_size=0.6,则test_size默认为0.4。

                   -- train_size:训练样本占比。

　    　random_state：是随机数的种子。

　　    随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
                比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，
                每次都会不一样。随机数的产生取决于种子，随机数和种子之间的关系遵从以下两个规
                则：种子不同，产生不同的随机数；种子相同，即使实例不同也产生相同的随机数。

'''
#3.训练svm分类器
classifier=svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovr') # ovr:一对多策略
classifier.fit(train_data,train_label.ravel()) #ravel函数在降维时默认是行序优先
'''
    kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。

　 kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。

　 decision_function_shape='ovr'时，为one v rest（一对多），即一个类别与其他类别进行划分，

　 decision_function_shape='ovo'时，为one v one（一对一），即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。

'''

#4.计算svc分类器的准确率
print("训练集：",classifier.score(train_data,train_label))
print("测试集：",classifier.score(test_data,test_label))

#也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score
tra_label=classifier.predict(train_data) #训练集的预测标签
tes_label=classifier.predict(test_data) #测试集的预测标签
print("训练集：", accuracy_score(train_label,tra_label) )
print("测试集：", accuracy_score(test_label,tes_label) )


test_y= classifier.predict(test_x)
test_y=test_y.reshape(-1,1)
label=np.concatenate([y,test_y],axis=0)
label.save('word2vec/svm_y')






