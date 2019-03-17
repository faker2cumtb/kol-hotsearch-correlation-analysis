# -*- coding: utf-8 -*-
import argparse

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import re


def format_content(weibo):
    """
    过滤特殊字符，对文本进行分字，对数字和英文进行分词
    :param weibo: JY2018解说是这样的形式
    :return: 2018 JY 解 说 是 这 样 的 形 式
    """
    new_weibo = []
    pattern = re.compile(u'[\u4e00-\u9fa5]')  ##匹配汉字
    pattern_digit = re.compile(u'[0-9]+')  ##匹配数字
    pattern_english = re.compile(u'[a-zA-Z]+')  ##匹配英文
    pattern_expression_http = re.compile(u'(http[^\u4e00-\u9fa5]+)|(\[.+?\])')  ##过滤表情字符[]和http字符
    weibo = re.sub(pattern_expression_http, '', weibo)
    ##正则匹配数字

    for item in re.finditer(pattern_digit, weibo):
        new_weibo.append(item.group())
    ##正则匹配英文
    for item in re.finditer(pattern_english, weibo):
        new_weibo.append(item.group().upper())
    ##分字
    for word in weibo:
        if re.match(pattern, word):
            new_weibo.append(word)
    return ' '.join(list(new_weibo))


def norm_cosine_distance(A, B):
    """
    计算两个向量的余弦距离，返回归一化的结果
    :param A: 向量A
    :param B: 向量B
    :return: 归一化的余弦距离
    """
    num = np.dot(A, B.T)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    return 0.5 + 0.5 * (num / denom)


def weight_factor(weibo_content, topic_content):
    """
    根据主题词在微博中出现的程度给出得分值的权重，
    具体计算为主题词在微博中出现的个数 除以 主题词的个数
    对于数字英文做特殊处理，如过主题词中的数字英文不在微博中出现，则分母 加上 数字英语的长度-1
    :param weibo_content: 微博分字内容
    :param topic_content: 主题分字内容
    :return: 得分值的权重
    """
    len_topic_in_weibo = 0
    len_topic_content = 0
    for i in topic_content:
        if i in weibo_content:
            len_topic_in_weibo += len(i)
        len_topic_content += len(i)
    return len_topic_in_weibo / len_topic_content


def score_function(nom_cosine, w):
    """
    最终得分值为归一化的余弦距离 * 权重项
    :param nom_cosine:
    :param w:
    :return:
    """
    return nom_cosine * w


def tf_idf(weibo_list, topic_list):
    """
    训练tf_idf模型
    :param weibo_list: weibo语料
    :param topic_list: topic语料
    :return: tf_idf模型
    """
    doc_list = weibo_list.copy()
    doc_list.extend(topic_list)
    doc_list = list(set(weibo_list))
    for i in range(len(doc_list)):
        doc_list[i] = format_content(doc_list[i])
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")  ##分词模式，加上这个防止把单字当停用词处理
    vectorizer.fit(doc_list)
    return vectorizer


if __name__ == "__main__":
    pred_label = []
    score_list = []
    parser = argparse.ArgumentParser("kol-hotsearch-correlation-analysis")
    """文件格式的说明见text_util
    """
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="classify threshold")
    parser.add_argument(
        "--data_file_path",
        type=str,
        required=True,
        help="The path of data. ")
    argments = args = parser.parse_args()
    threshold = argments.threshold
    coupus_path = argments.data_file_path
    # coupus_path = '/Users/Administrator/Desktop/social_analysis/luoxinyu_test/social_weibo_compare_hot_search_result_0901_0903.csv'
    DF = pd.read_csv(filepath_or_buffer=coupus_path, header=None)
    weibo_list = list(DF.iloc[:, 2])
    topic_list = list(DF.iloc[:, 4])
    true_lable = list(DF.iloc[:, 11])

    TF_IDF = tf_idf(weibo_list, topic_list)

    ##得到pred_label 和 score_list ，阈值
    for i in range(len(weibo_list)):
        weibo_array = TF_IDF.transform([format_content(weibo_list[i])]).toarray()[0]
        topic_array = TF_IDF.transform([format_content(topic_list[i])]).toarray()[0]
        nom_cosine = norm_cosine_distance(weibo_array, topic_array)
        weight = weight_factor(format_content(weibo_list[i]).split(), format_content(topic_list[i]).split())
        print(nom_cosine, weight)
        score = score_function(nom_cosine, weight)
        score_list.append(score)

        if (score >= threshold):
            pred_label.append(1)
        else:
            pred_label.append(0)

    print(classification_report(y_true=true_lable, y_pred=pred_label))
    print(confusion_matrix(y_true=true_lable, y_pred=pred_label))

    for i in range(len(true_lable)):
        ## FN
        # if label[i] == 1 and pred_labe[i] == 0:
        ## FP
        if true_lable[i] == 0 and pred_label[i] == 1:
            print(DF.iloc[i, 2], DF.iloc[i, 8], DF.iloc[i, 4])
            print(format_content(weibo_list[i]))
            print(score_list[i])
            print(format_content(topic_list[i]))
            print(i)

            print()
