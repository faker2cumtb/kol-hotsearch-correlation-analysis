# -*- coding: utf-8 -*-
## 记录结果
## 标注数据977个，热搜数据为9月1日至9月3日，共三天
## 设置阈值T大于等于T预测结果为1，小于T预测结果为0
## 默认阈值小于0.25的真实值都为0,
## 以下为跑出的结果
import json
import pandas as pd

T = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
dict = {
    0.25: {'FP': 509, 'TP': 468, 'FN': 0},
    0.3: {'FP': 245, 'TP': 385, 'FN': 83},
    0.35: {'FP': 120, 'TP': 318, 'FN': 150},
    0.4: {'FP': 55, 'TP': 263, 'FN': 205},
    0.45: {'FP': 25, 'TP': 202, 'FN': 266},
    0.5: {'FP': 9, 'TP': 155, 'FN': 313},
    0.55: {'FP': 2, 'TP': 100, 'FN': 368}
}


def count_FN(dict):
    """
    计算‘FN’值
    :param dict:
    :return:
    """
    for key in dict.keys():
        if key != 0.25:
            dict[key]['FN'] = dict[0.25]['TP'] - dict[key]['TP']
    return dict


def statistics(TP, FP, FN):
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    F1_score = 2 * Precision * Recall / (Precision + Recall)
    return {
        'Precision': Precision,
        'Recall': Recall,
        'F1_score': F1_score
    }


if __name__ == '__main__':
    list = [[]]
    for key in dict.keys():
        TP = dict[key]['TP']
        FP = dict[key]['FP']
        FN = dict[key]['FN']
        dict[key]['Precision'] = statistics(TP, FP, FN)['Precision']
        dict[key]['Recall'] = statistics(TP, FP, FN)['Recall']
        dict[key]['F1_score'] = statistics(TP, FP, FN)['F1_score']
    for key, value in dict.items():
        print(dict[key])
