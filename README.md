# kol-hotsearch-correlation-analysis
KOL所发微博内容与微博热搜的关联性分析二分类(相关,不相关)


[![license](https://img.shields.io/github/license/go88/fer2013-recognition.svg?style=for-the-badge)](https://choosealicense.com/licenses/mit/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=for-the-badge)](https://github.com/faker2cumtb/kol-hotsearch-correlation-analysis/pulls)
[![GitHub (pre-)release](https://img.shields.io/github/release/go88/fer2013-recognition/all.svg?style=for-the-badge)](https://github.com/faker2cumtb/kol-hotsearch-correlation-analysis/releases)

---

## 目录

```text
conf/       停用词典和和一些分词自定义词典
    dict/    分词自定义词典
data/     数据集
    kol_weibo/       kol微博数据
utils/      一些数据处理工具和统计工具
test.py         一些功能测试
weibo_topic_analysis.py         主程序

```

---
## 模型介绍
### 算法模型
基于字的TF-IDF，归一化余弦距离，热搜_微博权重系数
### 算法流程
1. 语料准备：语料去重，过滤特殊字符，过滤表情符号[],过滤http符号，英文小写转大小，对数字英文进行分词，汉字进行分字
2. 训练TF-IDF模型
3. 计算微博内容向量与热搜词向量的归一化余弦距离nom_cosine_distance
4. 计算热搜_微博权重系数w
5. 计算最终得分值：nom_cosine_distance * w


## 主程序介绍
weibo_topic_analysis.py 
   
终端运行 weibo_topic_analysis.py文件需要输入的相关参数介绍

| Parameter | Introduce | Demo |
| ------ | ------ | ------ |
|--threshold|分类阈值|0.6|
|--data_file_path|文件路径|


## python及相关依赖包版本
| Name | Version | 
| ------ | ------ | 
|python|3.6.7|
|PyMySql|0.9.3|
|jieba|0.39|
|gensim|3.7.0|

