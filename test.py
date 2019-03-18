# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import csv
import jieba
import os
from sklearn.metrics import classification_report, confusion_matrix
from utils.string_utils import format_weibo

dict_path = '/Users/Administrator/Desktop/social_analysis/conf/dict/user_dict.txt'
jieba.load_userdict(dict_path)
path = '/Users/Administrator/Desktop/social_analysis/luoxinyu_test/social_weibo_compare_hot_search_result_0_25_new.csv'
DF = pd.read_csv(filepath_or_buffer=path, header=None)
DF.rename(columns={2: 'weibo_content', 4: 'hot_search_topic', 8: 'score', 11: 'label'})
## 2微博内容，4热搜词，8得分值，11标签
print(DF.iloc[:10, [2, 4, 8, 11]])
new_label = []
print(DF.shape[0])
stopword_list = []


def func_strip(x):
    return x.strip()


with open('/Users/Administrator/Desktop/social_analysis/conf/stopwords.txt', 'r', encoding='utf8') as f:
    words = f.readlines()
    stopword_list = list(map(func_strip, words))
print(len(stopword_list))
if '出' in stopword_list:
    print('True')


## 不分词直接分字
def cut_by_word(content):
    return set([word for word in content])


def filter_stopwords(stopword_list, list):
    filter_list = []
    for word in list:
        if word in stopword_list:
            continue
        else:
            filter_list.append(word)
    return filter_list


def format_content(content_list):
    list = []
    for word in content_list:
        if word != " ":
            list.append(word)
    return list


c_weibo_list, c_topic_list = [], []
for i in range(DF.shape[0]):
    ## 分词
    # c_weibo = filter_stopwords(stopword_list,[word for word in jieba.cut(format_weibo(DF.iloc[i, 2]))])
    # c_topic = filter_stopwords(stopword_list,[word for word in jieba.cut(format_weibo(DF.iloc[i, 4]))])
    c_weibo = format_content(filter_stopwords(stopword_list, [word for word in jieba.cut(format_weibo(DF.iloc[i, 2]))]))
    c_topic = format_content(filter_stopwords(stopword_list, [word for word in jieba.cut(format_weibo(DF.iloc[i, 4]))]))
    ## 不分词，只考虑字
    # c_weibo = filter_stopwords(stopword_list,cut_by_word(format_weibo(DF.iloc[i, 2])))
    # c_topic = filter_stopwords(stopword_list,cut_by_word(format_weibo(DF.iloc[i, 4])))

    # c_weibo = cut_by_word(format_weibo(DF.iloc[i, 2]))
    # c_topic = cut_by_word(format_weibo(DF.iloc[i, 4]))
    c_weibo_list.append(c_weibo)
    c_topic_list.append(c_topic)
    if set(c_topic).difference(c_weibo):
        new_label.append(0)
    else:
        new_label.append(1)
label = list(DF.iloc[:, 11])
print(new_label)
print(label)
print(sum(np.equal(new_label, label)) / len(new_label))
print(classification_report(y_pred=new_label, y_true=label))
print(confusion_matrix(y_pred=new_label, y_true=label))
for i in range(len(label)):
    ## FN
    if label[i] == 1 and new_label[i] == 0:
        # if label[i] == 0 and new_label[i] == 1:
        print(DF.iloc[i, 2], DF.iloc[i, 8], DF.iloc[i, 4])
        print(c_weibo_list[i])
        print(DF.iloc[i, 8])
        print(c_topic_list[i])
        print(i)

        print()

from sklearn.feature_extraction.text import TfidfVectorizer
import re
from gensim import corpora, models, similarities


def fomat_content(stopword_list, weibo):
    """
    只保留汉字英文数字
    :param stopword_list:
    :param weibo:
    :return: 返回一个列表
    """
    new_weibo = []
    stopword_list = []
    pattern = re.compile(u'[\u4e00-\u9fa5]')
    pattern_digit = re.compile(u'[0-9]+')
    pattern_english = re.compile(u'[a-zA-Z]+')
    for item in re.finditer(pattern_digit, weibo):
        new_weibo.append(item.group())
    for item in re.finditer(pattern_english, weibo):
        new_weibo.append(item.group().upper())
    for word in weibo:
        if re.match(pattern, word) and (word not in stopword_list):
            new_weibo.append(word)
    return ' '.join(list(new_weibo))


def func_filter(content):
    fomat_content(content)


import numpy as np


def cosine_distance(A, B):
    """
    计算标准化余弦距离
    :param A: 向量A
    :param B: 向量B
    :return: 距离
    """
    # print (B.T)
    # print (A)
    # num = float(A * B.T)
    num = np.dot(A, B.T)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    return 0.5 + 0.5 * (num / denom)


def func_score(score, weibo_content, topic_content):
    len_topic_in_content = 0
    len_topic_content = len(topic_content)
    for i in topic_content:
        if i in weibo_content:
            len_topic_in_content += 1
        else:
            if len(i) > 1:
                len_topic_content += (len(i) - 1)

    a = len_topic_in_content / len_topic_content
    return score * a


def tf_idf_word(DF):
    """
    tf_idf向量化
    :param DF: 数据dataframe
    :return:
    """
    weibo_list = list(DF.iloc[:, 2])
    topic_list = list(DF.iloc[:, 4])
    print(len(weibo_list))
    print(len(topic_list))
    doc_list = weibo_list.copy()
    doc_list.extend(topic_list)
    doc_list = list(set(weibo_list))
    for i in range(len(doc_list)):
        doc_list[i] = fomat_content(stopword_list, doc_list[i])
    # # 生成词典tfidf[corpus]
    # dictionary = corpora.Dictionary(doc_list)
    # # 词袋模型
    # corpus = [dictionary.doc2bow(doc) for doc in doc_list]
    # # TF/IDF模型
    # tfidf = models.TfidfModel(corpus)
    #
    # # 相似度矩阵
    # index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
    vectorizer.fit(doc_list)
    sparse_result = vectorizer.transform(doc_list)
    # print (tfidf_model.vocabulary_)
    # print(sparse_result.todense())
    pred_labe = []
    score_list = []
    for i in range(len(weibo_list)):
        a = vectorizer.transform([fomat_content(stopword_list, weibo_list[i])]).toarray()[0]
        b = vectorizer.transform([fomat_content(stopword_list, topic_list[i])]).toarray()[0]
        score = func_score(cosine_distance(a, b), fomat_content(stopword_list, weibo_list[i]).split(),
                           fomat_content(stopword_list, topic_list[i]).split())
        score_list.append(score)
        if (score >= 0.6):
            pred_labe.append(1)
        else:
            pred_labe.append(0)
    print(classification_report(y_true=label, y_pred=pred_labe))
    print(confusion_matrix(y_true=label, y_pred=pred_labe))
    # print(len(weibo_list))
    # print (vectorizer.transform([fomat_content(weibo_list[977])]).toarray()[0])
    for i in range(len(label)):
        ## FN
        # if label[i] == 1 and pred_labe[i] == 0:
        if label[i] == 0 and pred_labe[i] == 1:
            print(DF.iloc[i, 2], DF.iloc[i, 8], DF.iloc[i, 4])
            print(fomat_content(stopword_list, weibo_list[i]))
            print(score_list[i])
            print(fomat_content(stopword_list, topic_list[i]))
            print(i)

            print()


if __name__ == '__main__':
    # 测试
    tf_idf_word(DF)
    # print (cosine_distance(np.array([0.1,0.2,0.3,0.4]),np.array([0.2,0.3,0.4,0.5])))
    # print([word for word in jieba.cut('亚运会闭幕式IKON')])
