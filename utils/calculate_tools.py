# -*- coding: utf-8 -*-
import numpy as np


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
    具体计算为主题词在微博中出现的个数（字符串长度） 除以 主题词的个数（字符串长度）

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
