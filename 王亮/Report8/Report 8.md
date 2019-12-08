# Report 8

> Week 14
>
> 08, Dec, 2019
>
> Liang Wang

## 1. read paper

---

**《Online Anomaly Detection on the Webscope S5 Dataset: A Comparative Study》**



### Abstract

**adaptability** is required

> adaptability的含义：
>
> 1. 适用于不同特征的数据
> 2. 不只用于离线的检测，还能进行online的检测

general idea: a significant deviation

SORAD is quite successful



### Introduction

"anomaly" is difficult to define. depends largely on the **context**

Difficult to benchmark. This paper just consider Yahho's well-known dataset.

Purpose: 

1. benchmarking anomaly detection algorithms is difﬁcult with currently well-established benchmark datasets.

2. SORAD show good performance on yahoo webscope s5



### Methods

A. Feature Generation using Sliding Windows

特征工程？

特征向量的格式：不是简单的滑窗，而是为每个滑窗位置生成一个特征向量：a bias item + L previous instance（倒序）
$$
\vec{x}_{k}=\left(1, y_{k}, y_{k-1}, \dots, y_{k-\ell+1}\right)^{T}
$$

> 这种方式是否适用于C-LSTM？

负索引的处理方式：用序列首项 $y_0$ 进行填充

B, C, D 是逐步演进的三个传统机器学习异常检测算法，没有详细阅读



### Experimental Setup

A. Yahoo’s S5 Webscope Dataset

The Webscope S5 dataset is a relatively large anomaly detection benchmark which is publicly available [1]. It consists of 367 time series, each of length 1500, in four different classes A1/A2/A3/A4 with class counts 67/100/100/100. While class A1 has real data from computational services, classes A2, A3, and A4 contain synthetic anomaly data with increasing complexity. Example plots are shown in Figs. 2–5.

A1: 真实数据

A2～A4: 合成数据，且合成的复杂程度越来越高。A2、A3只有离群点异常，A4中还加入了变化点异常。



正常数据的基本特征：稳定的周期性特征+噪声（周期性特征可能是多个周期函数的叠加）

异常数据：离群点+变化点。离群点也是对于context而言，不一定是数值过大过小，而是偏离了本应遵循的周期性规律。变化点可能是更高级的异常，更难被检测。



C. Algorithm Performance Measures

1. F1 score
    $$
    \begin{aligned} \text { precision } &=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}} \\ \text { recall } &=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}} \end{aligned}
    $$

    $$
    F 1=2 \cdot \frac{\text { precision } \cdot \text { recall }}{\text { precision }+\text { recall }}
    $$

2. computation time



D. Anomaly Windows

定义了Anomaly Widow的概念：在滑窗长度为10的情况下，Anomaly Window 的范围是从异常点加上前后各5个步长的范围

根据上述概念来判定TP, FP, TN, FN

> 感觉此处不太合理，对于异常点之前的实例，此时异常还没有出现，为什么就要认为是异常序列呢

---

### 总结：

这篇文章的工作体现在两个方面：一是说明了对于异常检测问题，adaptability是个难题，很难去进行benchmark；二是对几个算法进行了实验，提出了在webscope S5数据集上表现很好的SORAD算法。

对于我们的科研项目而言，我认为主要有以下收获：

1. 对于异常检测问题，adaptability是目前仍未得到有效解决的问题。所谓的"异常"，主要依赖于上下文，所以没有对于各种数据集普适性很强的算法。

    > 见 Introduction 部分

2. 加深了对Yahoo webscope s5数据集的理解

    > 见 Experimental Setup -> A 部分

3. 学到了一种针对上述数据集进行特征工程和性能评价的方法

    > 见 Method -> A 和 Experimental Setup -> C, D 部分



## 2. Continue Experiment

`train_gan.py` 源码阅读

过程：

加载配置

预训练



疑问：

预训练过程里用到的 clstm_classifier 的作用？