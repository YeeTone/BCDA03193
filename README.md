# BCDA03193餐饮服务评价情感倾向分析实训报告

## 1.1 背景与挖掘目标

本案例以国内不同餐饮企业的餐品评论数据为数据源，获取了餐饮服务中的评论数据与评价标签， 并对获取的数据进行了数据预处理和文本情感分析与挖掘。
本案例将不同分类器模型的实现的准确率进行了横向比较，并训练了一个餐饮评论情感倾向的MLP分类器模型。
本案例从多个维度评估了该分类器的分类效果，最终验证该分类器的分类效果是比较理想的。

## 1.2 数据预处理

### 1.2.1 文件读取

首先将评论数据由本地文件读取到内存中，代码如下所示：

```python
import pandas as pd

data = pd.read_excel('../data/data.xlsx') # 读取文件
data.head()
```

### 1.2.2 文本数据预处理

读取出来的数据包含4列，target（评价标签），userId（用户id），sellerId（商家id），timestamp（评价时间戳）和comment（评价内容）。

前四个数据无需清洗处理，而读取出的文本数据会包含"text："字符串，需要去除处理；另外用户的评论列会包含一些其他的字符，也需要去除处理。

```python
import re

data['comment'] = data['comment'].apply(lambda x: x.replace('text：','')) # 去除text：
data['comment'] = data['comment'].apply(lambda x: 
        re.sub('[^\u4E00-\u9FD5,.?!，。！？、；;:：0-9]+', '', x)) 
# 去除非中英文和数字的其他符号
data.head()
```

## 1.3 评价情绪统计分析

### 1.3.1 评价情绪归类统计

将数据中的情绪标签进行归类并统计结果，实现代码如下：

```python
import matplotlib.pyplot as plt

def make_autopct(values): # 让饼图自动显示数值和比例
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}% ({v:d})'.format(p=pct,v=val)
    return my_autopct

num = data['target'].apply(lambda x: '积极' if x == 0 else '消极').value_counts()
# 设置标签并统计数量
plt.figure(figsize=(4,4))
plt.rcParams['font.sans-serif'] = 'Simhei' # 避免中文乱码
plt.pie(num, autopct=make_autopct(num), labels=num.index)
plt.title('餐品积极/消极评论标签') # 设置标题
plt.show()
```

![image](https://user-images.githubusercontent.com/64548919/210151257-319423ad-5f2e-4a82-a9c1-02c7f04a92ef.png)

评价统计结果：

| 评价情绪 | 评价数量 |
|----------|------|
| 积极     | 8672 |
| 消极     | 9281 |

评价分析：

从饼图和表格中我们可以看到用户对餐品的积极评价和消极评价数量接近，其中消极评价稍多一些。

### 1.3.2 分词与去除停词

接下来对清洗后按用户情绪分类的评论进行文本分词以及去除停用词。
文本分词使用的是中文分词库jieba，停用词使用了github上开源的停用词表。
实现代码如下：

```python
import jieba

with open('../stopword/stopword-cn.txt','r', encoding = 'utf-8') as f:
    stopwords = f.read()
    
stopwords = stopwords.split() # 将读取的停词数据分割为列表
stopwords.append(' ') # 将空格和换行符添加至停词字典
stopwords.append('\n')

data_neg = data[data['target'] == 1] # 消极评价
data_pos = data[data['target'] == 0] # 积极评价

data_neg_cut = data_neg['comment'].apply(jieba.lcut) # 应用jieba分词
data_neg_cut = data_neg_cut.apply(lambda x : 
                [i for i in x if i not in stopwords]) # 去除停词
print(data_neg_cut.head())

data_pos_cut = data_pos['comment'].apply(jieba.lcut) # 应用jieba分词
data_pos_cut = data_pos_cut.apply(lambda x : 
                [i for i in x if i not in stopwords]) # 去除停词
print(data_pos_cut.head())
```

### 1.3.3 统计词频与词云图绘制

对去除完停用词后的数据进行词频统计并绘制词云，分析一下所有商家的用户评论中最频繁讨论的词主要是哪些。实现代码如下：

```python
from wordcloud import WordCloud
from PIL import Image

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

def show(wc, fn=None):
    # wc 词云图
    # fn 目标输出的文件名
    # 显示词云图，如果有需要可以输出到本地文件
    
    plt.imshow(wc)
    plt.axis('off')
    plt.show()
    if fn is not None:
        wc.to_file(fn)

freq = pd.Series(list(itertools.chain(*list(data_pos_cut)))).value_counts()
# 词出现的频次
mask = np.array(Image.open('../stopword/China.jpg'))
# 加载词云图模型
wc = WordCloud(scale=4,
               width=2500, 
               height=3000,
               font_path='C:/Windows/Fonts/simkai.ttf',
               background_color='White', mask=mask)
    # 构建词云图模型
wc2 = wc.fit_words(freq) # 将词云图模型应用至字典
show(wc2, '../stopword/wordcloud.png')
```

![image](https://user-images.githubusercontent.com/64548919/210166114-80bb469e-b5d6-4445-adfe-9cb40b37a1a3.png)

同样可以对积极评价和消极评价分别进行词云图的绘制与分析。具体实现代码如下：

```python
import pandas as pd
import itertools

freq_pos = pd.Series(lst(itertools.chain(*list(data_pos_cut)))).value_counts()
# 积极词频统计
freq_neg = pd.Series(list(itertools.chain(*list(data_neg_cut)))).value_counts()
# 消极词频统计

wc2_pos = wc.fit_words(freq_pos) # 将词云图模型应用至字典
show(wc2_pos, '../stopword/wordcloud_pos.png')

wc2_neg = wc.fit_words(freq_neg) # 将词云图模型应用至字典
show(wc2_neg, '../stopword/wordcloud_neg.png')
```

![wordcloud_pos](https://user-images.githubusercontent.com/64548919/210166141-51bac063-85f1-4c1f-b1f8-41c137143d81.png)

![wordcloud_neg](https://user-images.githubusercontent.com/64548919/210166142-9092f388-1408-4865-ae50-64875300f25f.png)

对积极评价和消极评价的前十词频进行筛选，统计结果如下：

- 积极评价：

| 排名 | 单词  | 词频   |
|------|-----|------|
| 1    | 不错  | 3618 |
| 2    | 吃   | 2988 |
| 3    | 味道  | 2373 |
| 4    | 好吃  | 2069 |
| 5    | 喜欢  | 1496 |
| 6    | 比较  | 979  |
| 7    | 环境  | 932  |
| 8    | 感觉  | 812  |
| 9    | 挺   | 745  |
| 10   | 服务  | 676  |

- 消极评价：

| 排名 | 单词  | 词频   |
|------|-----|------|
| 1    | 吃   | 2223 |
| 2    | 真的  | 2005 |
| 3    | 味道  | 1264 |
| 4    | 没   | 1239 |
| 5    | 好吃  | 1039 |
| 6    | 小时  | 960  |
| 7    | 说   | 926  |
| 8    | 点   | 912  |
| 9    | 难吃  | 905  |
| 10   | 送   | 864  |

我们从积极评价和消极评价的词云图中可以归纳共同点：用户对于”吃”的关注度是最高的，并且非常重视食物的味道。
积极评价和消极评价的主要不同点集中在对于用餐体验的形容上：积极评价会有更多的正面形容词出现在词云图中，而消极评价会有更多负面的形容词出现在词云图中。

## 1.4 用户情绪与评价时间分析统计

### 1.4.1 用户情绪随日期的变化

这部分分析随着日期的变化，用户对餐品评价的数量变化情况。实现代码如下：

```python
import matplotlib.pyplot as plt

comm_day_pos = data_pos['timestamp'].apply(lambda x: x.strftime('%Y-%m')).value_counts()
comm_day_pos = comm_day_pos.sort_index()
comm_day_neg = data_neg['timestamp'].apply(lambda x: x.strftime('%Y-%m')).value_counts()
comm_day_neg = comm_day_neg.sort_index()
# 将日期格式转化为年月形式，并排序

plt.figure(figsize=(8,4))
plt.plot(range(len(comm_day_pos)), comm_day_pos, label='积极情绪评价')
plt.plot(range(len(comm_day_neg)), comm_day_neg, label='消极情绪评价')
plt.xticks(range(len(comm_day_pos)), comm_day_pos.index,rotation=45)
# 将横坐标文本旋转45度，使展示更美观
plt.grid()
plt.title('积极/消极评价随日期变化图')
plt.xlabel('日期')
plt.ylabel('用户评价数量')
plt.legend()
```

![image](https://user-images.githubusercontent.com/64548919/210165288-6397a696-db9c-4dcf-83b7-86386fbd532d.png)

由于只有2020年9月的月初的数据，因此2020年9月的数据显示了一个低谷。

我们可以看到积极/消极情绪评价的曲线并不完全相似。这里涉及2个重要时间点：2019年12月新冠疫情爆发、2020年3月全国企业复工。
我们可以看到新冠疫情爆发后餐饮行业出现了积极评价的减少和消极评价的增加，并且消极评价在2020年1月达到了峰值。
全国企业复工后也带来了积极评价的减少和消极评价的增加，并且消极评价在2020年5月达到峰值。

疫情爆发后出现的积极评价减少和消极评价增加是符合常识的，因为疫情爆发对餐饮行业带来了比较大的冲击力，导致很多餐饮行业不能正常运转。

复工前消极评价出现了一次低谷，说明复工前人们逐渐适应了疫情下的餐饮行业。
复工后由于餐饮行业还没有完全恢复至疫情前的水平，我们可以看到人们此时对餐饮行业的当时情况并不满意。
从这些数据分析结果来看，我们可以认识到疫情形势一方面影响了用户本身的点餐用餐体验，另一方面对餐饮业带来了巨大的冲击，且恢复起来需要数月的时间。

### 1.4.2 用户情绪随时刻的变化

接下来分析用户的积极情绪评价和消极情绪评价随时刻的变化，并将二者放入同一图表中做对比。实现代码如下：

```python
import pandas as pd
import matplotlib.pyplot as plt

comm_hour_pos = pd.to_datetime(data_pos['timestamp']).apply(lambda x: x.hour).value_counts()
comm_hour_pos = comm_hour_pos.sort_index()

comm_hour_neg = pd.to_datetime(data_neg['timestamp']).apply(lambda x: x.hour).value_counts()
comm_hour_neg = comm_hour_neg.sort_index()
# 将日期格式转化为年月形式，并排序

plt.plot(comm_hour_pos.index, comm_hour_pos, label='积极情绪评价')
plt.plot(comm_hour_neg.index, comm_hour_neg, label='消极情绪评价')
# 同时绘制积极情绪和消极情绪的折线图
plt.title('积极/消极评价随时刻变化图')
plt.xticks(ticks=range(0,24))
plt.yticks(ticks=range(0,1000,100)) # 设置x,y轴的范围
plt.xlabel('时刻')
plt.ylabel('用户评价数量') # 设置x,y轴的标签
plt.grid() # 设置网格化
plt.legend()
```

![image](https://user-images.githubusercontent.com/64548919/210165233-3ab464a2-619b-42cd-8921-19f098b22531.png)

我们可以看到积极情绪评价和消极情绪评价曲线基本上接近，并且峰值出现在中午11-13时以及晚上19-20时。
这主要是由用户的用餐规律所决定的，大多数用户更有可能在午餐或者晚餐的时候订餐。

同时餐饮评价一般是在用户用完餐后发布的，因此用户更偏向于食用后再进行对餐品的评价行为。

## 1.5 积极情绪最多的商家与优点分析

### 1.5.1 积极评价最多商家
对分类后的积极评价商家进行计数归类，得到积极评价最多的商家是sellerId为1041的商家，有54条积极评价，具体代码如下：

```python
import matplotlib.pyplot as plt

def autolabel(rects): # 设置柱状图的数值显示
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.3, 1.03*height, '%s' % int(height), size=18, family="Consolas")

best_sellers = data_pos['sellerId'].value_counts().nlargest(10) # 选取积极评价最多的前十名商家
autolabel(plt.bar(range(len(best_sellers[:10])), best_sellers[:10], label='积极评论数量'))
plt.xticks(range(len(best_sellers[:10])), best_sellers[:10].index, rotation=45)
plt.title('积极评论最多商家') 
plt.xlabel('商家Id')
plt.ylabel('积极评论数量') # 设置x轴和y轴的标签
plt.grid() # 设置网格化
plt.legend()
```

![image](https://user-images.githubusercontent.com/64548919/210336202-3d4306cf-8bf1-4099-9f2b-ced5c767c833.png)

从上图我们可以看出1041号商家排名第一，并且远超第二名和第三名。对于该现象的分析，我们将在1.5.2中进行详细讨论。

### 1.5.2 积极评价最多商家优点分析

收集积极评价最多商家（1041号）的所有评价，并利用jieba分词和中文停词库等制作词云图，代码如下所示：

```python
import pandas as pd
import itertools
import jieba

best_seller_comments = data_pos[data_pos['sellerId'] == 1041]['comment'] # 选取1041号商家的评论
best_seller_comments_cut = best_seller_comments.apply(jieba.lcut) # 分词
best_seller_comments_cut = best_seller_comments_cut.apply(lambda x : 
                                [i for i in x if i not in stopwords]) # 去除停词
best_seller_comments_freq = pd.Series(list(itertools.chain(*list(best_seller_comments_cut)))).value_counts() # 计算词频
best_seller_comments_wc2 = wc.fit_words(best_seller_comments_freq) # 设置词云图
show(best_seller_comments_wc2, '../stopword/wordcloud_best_seller.png')
```

![wordcloud_best_seller](https://user-images.githubusercontent.com/64548919/210170699-bd017367-d138-451d-a5ca-36dd691fc6e6.png)

与之类似，可以制作第二名（sellerId为22024）和第三名（sellerId为20877）的评价词云图：

- 积极评论数第二名：

![wordcloud_best_seller](https://user-images.githubusercontent.com/64548919/210170784-95284ef2-bc2a-41a9-9a8a-7fc05d12a2a7.png)

- 积极评论数第三名：

![wordcloud_best_seller](https://user-images.githubusercontent.com/64548919/210170819-94b600e0-2dc6-429e-8322-af4c1e17cb30.png)

我们将第二名和第三名的词云图与第一名进行比较，我们可以发现第一名的词云主要集中于”味道不错”，”实惠”，”速度”，”好吃”，”满意”等形容词字眼。
而第二名和第三名的词云则更多集中于具体的餐品对象，如”奶油”，”栗子”，”云吞”，”粥”等等。第一名更多用形容词字眼，说明该商家食品的高质量是普遍的，更容易收到消费者的青睐，因此第一名的积极评论会远多于第二名和第三名。

通过比较分析，我们得出积极评价最多商家的优点主要是餐品质量高，配送快，价格实惠，令人满意。这符合我们对优质餐品与商家的认知。

## 1.6 消极情绪最多商家与改进建议

### 1.6.1 消极评价最多商家

对分类后的消极评价商家进行计数归类，得到消极评价最多的商家是sellerId为971的商家，有44条消极评价，具体代码如下：

```python
import matplotlib.pyplot as plt

def autolabel(rects): # 设置柱状图的数字显示
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.3, 1.03*height, '%s' % int(height), size=18, family="Consolas")

worst_sellers = data_neg['sellerId'].value_counts().nlargest(10)
# 选取消极评论最多的前十名商家
autolabel(plt.bar(range(len(worst_sellers[:10])), worst_sellers[:10], label='消极评论数量'))
# 绘制柱状图
plt.xticks(range(len(worst_sellers[:10])), worst_sellers[:10].index, rotation=45)
# 横轴字体旋转45度，更美观
plt.title('消极评论最多商家') # 设置标题
plt.xlabel('商家Id')
plt.ylabel('积极评论数量') # 设置x,y轴标签
plt.grid() # 设置网格
plt.legend()
```

![image](https://user-images.githubusercontent.com/64548919/210336140-6f2ce6af-92fa-4b33-9d53-c697857a264c.png)

可以看到971号商家的消极评论远多于其他的商家，具体的原因与改进策略会在1.6.2中做详细分析与讨论。

### 1.6.2 消极评价最多商家原因分析与改进建议

收集消极评价最多商家（971号）的所有评价，并利用jieba分词和中文停词库等制作词云图，代码如下所示：

```python
import pandas as pd
import jieba
import itertools

worst_seller_comments = data_neg[data_neg['sellerId'] == 971]['comment']
# 选取971号商家的评论
worst_seller_comments_cut = worst_seller_comments.apply(jieba.lcut)
# jieba分词
worst_seller_comments_cut = worst_seller_comments_cut.apply(lambda x : [i for i in x if i not in stopwords])
# 去除停词
worst_seller_comments_freq = pd.Series(list(itertools.chain(*list(worst_seller_comments_cut)))).value_counts()
# 计数统计
worst_seller_comments_wc2 = wc.fit_words(worst_seller_comments_freq)
# 设置词云图
show(worst_seller_comments_wc2, '../stopword/wordcloud_worst_seller.png')
```

![image](https://user-images.githubusercontent.com/64548919/210171249-8b69f5c0-5f21-4ed3-9814-a7ded9b8cefd.png)

与之类似，我们也可以制作消极评论排名第二名（1173号）和第三名（872号）的商家的词云图：

- 消极评论数第二名：

![image](https://user-images.githubusercontent.com/64548919/210171333-6c34f25e-9119-43a2-b6a2-c6ff76d4f370.png)

- 消极评论数第三名：

![wordcloud_worst_seller](https://user-images.githubusercontent.com/64548919/210171359-89f6c755-9bee-4f55-8e67-57706bb37960.png)

我们将第二名和第三名的词云图与第一名进行比较，我们可以发现三者的负面评价词云分布都比较相似，如第一名集中在”太慢”，”差”，”少”，”凉”等
、第二名集中在”差”，”卫生”，”便宜”等、第三名集中在”差”，”外卖”，”送”等等。
三者综合比较，我们可以看到负面评价多的商家一般都有”差”的形容评价，而且鉴于当时的疫情形势，更多用户选择了外卖作为点餐形式，外卖的质量也会对用户的评价产生重大影响。

通过比较分析，我们得出消极评价最多商家存在的问题主要是两个方面：
- 餐品质量方面餐品质量差，份额少，食品质量让用户感到不适；
- 外卖配送方面配送速度较慢，且配送员的态度有待改善。

基于以上消极评论，我提出的改进建议主要是以下方面：

- 餐品质量方面：商家提高餐品的质量，增加餐品的份额，让用户食用餐品时能够改善评价
- 外卖配送方面：商家选择配送员时应优先选择配送速度快，服务态度好的配送员

## 1.7 情感倾向模型与评估

### 1.7.1 构建数据集

这步需要将原始的数据集进行筛选和合并，选择出评论列和表示情感的标签列，其中积极评论的标签列为0，消极评论的标签列为1。实现代码如下：

```python
import pandas as pd

data_new_pos = pd.DataFrame()
data_new_pos['comment'] = data_pos['comment'].apply(lambda x: str.join('', x)) # 分词合并
data_new_pos['label'] = 0 # 设置积极标签
data_new_pos.reset_index(inplace=True,drop=True) # 重设从0开始的索引

data_new_neg = pd.DataFrame() 
data_new_neg['comment'] = data_neg['comment'].apply(lambda x: str.join('', x)) # 分词合并
data_new_neg['label'] = 1 # 设置消极标签
data_new_neg.reset_index(inplace=True,drop=True) # 重设从0开始的索引

data_new = pd.concat([data_new_pos,data_new_neg],axis=0) # 评论合并
data_new.head()
```

### 1.7.2 文本向量化

由于原始的文本词义不够明确，因此需要将文本转化为计算机能够处理的形式，即分词与文本数字化。这里使用了词袋模型来完成文本的向量化，具体实现代码实现如下所示：

```python
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
data_new['comment'] = data_new['comment'].apply(jieba.lcut)
# jieba分词
data_new['comment'] = data_new['comment'].apply(lambda x: str.join(' ', x))
# 分词以空格为间隔进行合并，保留语义信息
X_comments = vect.fit_transform(data_new['comment']).toarray()
# X_comments = X_comments.toarray()
word_bag = vect.vocabulary_ # 词袋模型中各词所占的权重
```

### 1.7.3 数据集划分

将原始数据划分为训练集和测试集，比例为4:1，并且加入随机化因素排除数据选取带来的影响。

```python
from sklearn.model_selection import train_test_split

X_comments = vect.fit_transform(data_new['comment']).toarray()
Y_label = data_new['label']
test_ratio = 0.2
X_train, X_test, Y_train, Y_test = train_test_split(X_comments, Y_label, test_size=test_ratio, random_state=1)
# 划分训练集和测试集
```

### 1.7.4 模型选取与比较

这个问题中尝试了六种模型（MLP神经网络，随机森林，逻辑回归，SVM支撑向量机，朴素贝叶斯，KNN近邻分类）并进行了横向比较，以在测试集上的准确率为衡量指标，并作出模型类型与准确率的柱状图。
最终MLP多层神经网络是最佳模型，横向比较与准确率评估的实现代码如下：

```python
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

# 分类器初始化
mlp = MLPClassifier(max_iter = 3)
random_forest = RandomForestClassifier()
logistic = LogisticRegression()
svm = SVC()
bayes = GaussianNB()
knn = KNeighborsClassifier()

# 分类器训练
mlp.fit(X_train, Y_train)
random_forest.fit(X_train, Y_train)
logistic.fit(X_train, Y_train)
svm.fit(X_train, Y_train)
bayes.fit(X_train, Y_train)
knn.fit(X_train, Y_train)

# 分类器预测
Y_pred_random_forest = random_forest.predict(X_test)
Y_pred_mlp = mlp.predict(X_test)
Y_pred_logistic = logistic.predict(X_test)
Y_pred_svm = svm.predict(X_test)
Y_pred_bayes = bayes.predict(X_test)
Y_pred_knn = knn.predict(X_test)

# 计算准确率并制作相关图表
module_names = ['MLP神经网络', '随机森林', '逻辑回归', 'SVM支撑向量机', '朴素贝叶斯', 'KNN近邻']
acc_scores = [accuracy_score(Y_pred_random_forest, Y_test), 
              accuracy_score(Y_pred_mlp, Y_test),
              accuracy_score(Y_pred_logistic, Y_test),
              accuracy_score(Y_pred_svm, Y_test),
              accuracy_score(Y_pred_bayes, Y_test),
              accuracy_score(Y_pred_knn, Y_test)]

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
for i in range(len(x_data)):
    autolabel(plt.bar(x_data[i], y_data[i]))
plt.title("模型种类与准确率柱状图")
plt.xlabel("模型类型")
plt.ylabel("准确率")
plt.ylimit(0.6, 1)
plt.show()
```

![image](https://user-images.githubusercontent.com/64548919/210293275-94ff920e-6a7d-43d2-9cd1-1da5d4fce678.png)

从上图我们可以看到MLP神经网络以在测试集上92.3%的准确率成为解决情感倾向分类问题的最佳模型。因此我们接下来的部分都将会使用MLP神经网络为模型进行性能和误差的评测。

### 1.7.5 模型性能与误差评估

为了对建立的MLP分类器模型进行性能与误差评估，我们建立混淆矩阵，并计算相应指标完成模型的评估。混淆矩阵定义如下：

|         | 预测| 预测   |
|---------|---------|------|
|         | 预测为1 | 预测为0 |
| 预测为1 | TP      | FN   |
| 预测为0 | FP      | TN   |

代码实现如下：

```python
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

y_pred = Y_pred.astype(np.int_)
y_test = Y_test.astype(np.int_)
tp = sum(y_pred & y_test) # 计算
fp = sum((y_pred == 1) & (y_test == 0))
tn = sum((y_pred == 0) & (y_test == 0))
fn = sum((y_pred == 0) & (y_test == 1))
print('TP = %s, FP = %s, TN = %s, FN = %s' % (tp, fp, tn, fn))

cm = confusion_matrix(y_test, y_pred) # 绘制混淆矩阵
cm_display = ConfusionMatrixDisplay(cm).plot()
```

我们计算出TP = 836, FP = 87, TN = 782, FN = 91，并得到绘制出的混淆矩阵：

![image](https://user-images.githubusercontent.com/64548919/210339527-f4b74ce0-d187-4288-ba99-521c9d298b5f.png)

然后我们利用这些数值从数个方面评估实现的MLP分类器的模型：

- 准确率：对于给定的测试集，模型正确分类的样本数与总样本数之比。

![image](https://user-images.githubusercontent.com/64548919/210318669-1565501b-892d-42c9-9449-a9a9933391d6.png)

- 精确率：对于给定测试集，分类模型将正类样本预测为正类的数量与将样本预测为正类的综述的比例

![image](https://user-images.githubusercontent.com/64548919/210318716-57d5b652-dd13-46cc-816c-68be4de586fd.png)

- 召回率：对于给定测试集，模型将正类样本分为正类的数量与，模型分类正确的数量的比值。

![image](https://user-images.githubusercontent.com/64548919/210318732-3d9d067a-a7a1-4f81-bb96-d4aeccdd3947.png)

- F1_score：综合衡量模型的召回率和精确度

![image](https://user-images.githubusercontent.com/64548919/210318747-337bfdff-c396-4220-8d80-1d4f33df4482.png)

- Matthews相关系数：综合混淆矩阵，度量真实值和预测值之间的相关性。该值在\[-1, 1\]之间，越靠近1则分类效果越好。

![image](https://user-images.githubusercontent.com/64548919/210318766-4ca6cac4-cc3e-4194-bee7-edbde93ed02a.png)

实现代码如下：

```python
from sklearn.metrics import accuracy_score
import math

# 计算评估指标
accuracy = accuracy_score(Y_pred, Y_test)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)
mcc = (tp*tn - fp*fn)/ math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))

print('Accuracy = %s, Precision = %s, Recall = %s, F1_score = %s, MCC = %s' %(accuracy, precision, recall, f1_score, mcc))
```

我们计算出

Accuracy = 0.923, Precision = 0.906, Recall = 0.902, F1_score = 0.904, MCC = 0.802

综合上面的数值来看，我们实现的MLP分类器具有比较好的分类效果。

### 1.7.6 模型附件评测

#### 1.7.6.1 数据读取，清洗与分词处理
附件中的数据也是需要清洗的。由于测试数据test.xlsx和原始数据data.xlsx的评论数据格式基本相同，因此可以使用相同的数据清洗与分词方法，代码如下所示：

```python
import pandas as pd
import jieba
import re

data_test = pd.read_excel('../data/test.xlsx') # 读取测试文件
comment_origin = data_test['comment'].copy() # 保存评论数据
data_test['comment'] = data_test['comment'].apply(lambda x: x.replace('text：',''))
data_test['comment'] = data_test['comment'].apply(lambda x: re.sub('[^\u4E00-\u9FD5,.?!，。！？、；;:：0-9]+', '', x)) # 数据清洗

data_test['comment'] = data_test['comment'].apply(jieba.lcut) # jieba分词
data_test['comment'] = data_test['comment'].apply(lambda x: str.join(' ', x)) # 分词后使用空格连接，保留语义信息
data_test.head()
```

#### 1.7.6.2 模型预测

使用先前训练好的MLP模型进行情感标签的预测，代码如下所示：

```python
X_comments_test = vect.transform(data_test['comment']).toarray()
Y_pred_test = mlp.predict(X_comments_test) # 模型预测
```

#### 1.7.6.3 模型结果输出到文件

得到输出数据后，需要将结果先输出到target列中，并复原之前的评论内容，再输出到文件中，代码如下所示：

```python
data_test['target'] = Y_pred_test
data_test['comment'] = comment_origin # 恢复评论内容
data_test.to_excel('../data/test_out.xlsx', index=False) # 输出到文件
```

代码运行完成后，即可在test_out.xlsx文件夹中找到带有情感标签的预测输出文件。

## 参考文献

[1]张膂. 基于餐饮评论的情感倾向性分析. Diss. 昆明理工大学, 2016.

[2]程康鑫. "基于LSTM与CNN的中文餐饮评论情感特征提取算法.".

[3]张婷婷. 面向餐馆评论的情感分析关键技术研究. Diss. 哈尔滨工业大学.

[4]杨博文. "基于餐饮业网络评论的消费者情感极性分析." 计算机系统应用 27.8(2018):7.

[5]田晓丽陈雪琼. "基于大众点评的国际品牌酒店餐饮服务质量影响因素研究." 内蒙古科技与经济 6(2022):81-83.
