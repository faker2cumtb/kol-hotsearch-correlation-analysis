# -*- coding: utf-8 -*-
"""
一些字符串处理的方法
"""

import re


def string_is_float(str_float):
    """
    判断一个字符串是否是浮点数的字符串，整型也不是浮点数
    :param str_float:
    :return:
    """

    is_float = True
    try:
        int(str_float)
        is_float = False
    except ValueError:
        try:
            float(str_float)
        except ValueError:
            is_float = False

    return is_float


def format_weibo(weibo):  # , after_segment=False):
    """
    对于话题进行格式化，统一成一样含义的话题
    :param weibo:
    # :param after_segment: 是否分词之后的微博
    :return:
    """
    # 微博话题还原
    # if after_segment:
    #     pattern = re.compile('#[^#]+#')
    #     new_weibo = re.sub(pattern, lambda x: x.group(0).replace(' ', ''), weibo)
    # else:

    # topic 去掉表情符号
    # pattern = re.compile('(\[[^\[\]]{1,10}\])+')
    # new_weibo = re.sub(pattern, '', weibo)
    # new_weibo = weibo
    # new_weibo = new_weibo.replace(' ','')
    # new_weibo = ""
    # pattern = re.compile(u'[\u4e00-\u9fa5]|[a-zA-Z0-9]')
    # for word in weibo:
    #     if re.match(pattern,word):
    #         new_weibo += word
    # print (new_weibo)
    new_weibo = " ".join(weibo.strip().upper().split())

    # return weibo
    return new_weibo


def format_content(weibo):
    """
    过滤特殊字符，过滤表情字符，过滤http:之类的，对文本进行分字，对数字和英文进行分词
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


if __name__ == '__main__':
    digits = '12'
    # print (format_weibo(u"【香蜜沉沉烬如霜58 59蓝光4K剧透】 古装神话剧#电视剧香蜜沉沉烬如霜#下集看点：锦觅发现润玉两面三刀，不想再嫁给他，然而润玉将其软禁，江苏卫视幸福剧场明晚19： 30继续播出 http://t.cn/RFCnidX 0.250857 香蜜沉沉烬如霜59"))
    # print(string_is_float(digits))
    patt = re.compile(u'(http[^\u4e00-\u9fa5]+)|(\[.+?\])')
    patt1 = re.compile(u'\[.+?\]')
    a = re.sub(patt, '', '[cry]哈哈http://t.cn/Ezpkika')
    print(a)
