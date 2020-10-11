import pandas as pd
import datetime


def user_data():
    # 把旅游不相关的话题清理前的数据
    user = pd.read_csv('data/original_data/blog_11w.csv',
                       # names=['id','name','funs','web_id','blog','date','repost','coment','thumb'],
                       # header=None,
                       sep='\t',
                       dtype={'id': int, 'web_id': int, 'date': str},
                       error_bad_lines=False
                       )
    print(user.head())
    # 有names他会自己截断后面的数，没有names时他会读取

    year = user.iloc[:, -4].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M").year)
    month = user.iloc[:, -4].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M").month)
    user = user.drop('date', axis=1)

    user = pd.concat((user, year, month), axis=1)

    user['blog_num'] = 1

    RCTB = user.groupby(['id', 'year', 'month'])['repost', 'coment', 'thumb', 'blog_num'].sum().reset_index()
    # 用来计算用户每月影响力，并且做归一化处理：
    RCTB['power'] = RCTB.apply(lambda x: x['col1'] + 2 * x['col2'], axis=1)

    RCTB.to_csv('data/user_data/user_RCTB.csv', header=0, sep=',', index_col=None)
    print(RCTB.head())


def blog_data():
    # 把旅游不相关的话题清理后,并且标上地区标签的数据
    blog = pd.read_csv('data/where_when_data/blog_11w.csv',
                       # names=['id'，'web_id','blog',‘topic’，'date','repost','coment','thumb','place'],
                       # header=None,
                       sep='\t',
                       dtype={'id': int, 'web_id': int, 'date': str},
                       error_bad_lines=False
                       )
    print(user.head())

    year = user.iloc[:, -4].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M").year)
    month = user.iloc[:, -4].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M").month)
    blog = pd.concat((blog, year, month), axis=1)

    RCT = user.groupby(['id', 'web_id', 'topic', 'year', 'month', 'place'])['repost', 'coment', 'thumb'].sum().reset_index()

    # 用来计算微博传播力，并且做归一化处理：
    RCT['blog_power'] = RCT.apply(lambda x: (1 + 0.625 * x['repost'] + 0.2385 * x['coment'] + 0.1365 * x['thumb']) / 0.625 * x['repost'] + 0.2385 * x['coment'] + 0.1365 * x['thumb'], axis=1)

    RCT.to_csv('data/user_data/blog_RCT.csv', header=0, sep=',', index_col=None)
    print(RCT.head())
    # id,web_id,blog,top,date,repost,coment,thumb,year,month


def hot_o_n():
    RCTB = pd.read_csv('data/user_data/user_RCTB.csv', header=0, sep=',')
    RCT = pd.read_csv('data/user_data/blog_RCT.csv', header=0, sep=',', index_col=None)

    # 计算每条微博的热度
    RCT['user_power'] = 0
    for i in range(RCT.shape[0]):
        id = RCT.loc[i, 'id']
        year = RCT.loc[i, 'year']
        month = RCT.loc[i, 'month']
        RCT.loc[i, 'user_power'] = RCTB[(RCTB['id'] == id) & (RCTB['year'] == year) & (RCTB['month'] == month), 'power']
    RCT['result'] = RCT.apply(lambda x: 0.75 * x['blog_power'] + 0.25 * x['user_power'], axis=1)

    RCT.to_csv('data/user_data/blog_result.csv', header=0, sep=',', index_col=None)
