# Report 7

## 1. 数据集

Computing System Data

## S5 - A Labeled Anomaly Detection Dataset, version 1.0(16M)

Automatic anomaly detection is critical in today's world where the sheer volume of data makes it impossible to tag outliers manually. The goal of this dataset is to benchmark your anomaly detection algorithm. The dataset consists of real and synthetic time-series with tagged anomaly points. The dataset tests the detection accuracy of various anomaly-types including **outliers** and **change-points**. The synthetic dataset consists of time-series with varying trend, noise and seasonality. The real dataset consists of time-series representing the metrics of various Yahoo services.

异常数据包括：离群值、变更点

Plot结果：`./plot_dataset.html`



## 2. 实验：基于GAN的异常检测

### Extend data loader:

```python
class LoadType(Enum):
    all = 1
    only_normal = 2
    only_abnormal = 3
```



### Generator:

```python
G_W1 = weight_var([10, 30], 'G_W1')
G_b1 = bias_var([30], 'G_B1')

G_W2 = weight_var([30, 60], 'G_W2')
G_b2 = bias_var([60], 'G_B2')

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob
```

input

fc(relu)

fc(sigmoid)



### Discriminator:

```python
D_W1 = weight_var([60, 30], 'D_W1')
D_b1 = bias_var([30], 'D_b1')

D_W2 = weight_var([30, 1], 'D_W2')
D_b2 = bias_var([1], 'D_b2')

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit
```

input

fc(relu)

fc(sigmoid)



### Threshold:

最终训练完的模型，所有正常数据输入discriminator，输出的所有值的最小值min(normal_data_score)



实际上应该设置正常数据的最小输出作为阈值，小于这个值的才是异常

因为损失函数最大，实际上是判别器的输出最小



实验结果：

0.10806951

Number of correctly classified data is:  (16, 1)
Number of wrongly classified data is:  (8540, 1)

没有分类能力。可能是由于网络过于简单，也可能需要换一种设置阈值的方法，或者换一种分类的评价函数。

所有输出：`./result.txt`



~~结果分析：~~

~~正常序列的判别器分数范围：~~

~~82357~~

~~[   16   336  2537 16118 31341 28846  2606   526    30     1]~~

~~[0.00019427613924742282, 0.004079798924195879, 0.03080491032941948, 0.19570892577437254, 0.3805505300095924, 0.3502555945456974, 0.03164272617992399, 0.006386828077759025, 0.0003642677610889178, 1.2142258702963926e-05]~~



~~异常序列的判别器分数范围：~~

~~8556~~

~~[ 125  638 1103 1996 2550 1516  497  112   19    0]~~

~~[0.0146096306685367, 0.07456755493221132, 0.12891538101916783, 0.23328658251519402, 0.2980364656381487, 0.1771856007480131, 0.05808789153810192, 0.013090229079008883, 0.0022206638616175784, 0.0]~~



如果取平均值作为阈值：

0.5151243
Number of correctly classified data is:  (5682, 1)
Number of wrongly classified data is:  (2874, 1)

准确率：66%

