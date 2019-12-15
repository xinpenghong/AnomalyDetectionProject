

1. 预训练过程使用全部数据，对抗训练过程只使用正常数据
2. 对抗训练过程只用正常数据
3. 测试对抗训练结果用全部数据





**预训练过程：**

```
step: 4100, loss: 0.183, accuracy: 0.951, precision: 1, recall: 0.07, f1: 0.1
```

precision高，recall低（检准率高，检全率低），95%的accuracy接近样本不均衡的比例

说明模型可能只是单纯输出0类别



Keep_prob：每个元素被保留的概率，那么 keep_prob:1就是所有元素全部保留的意思。
一般在大量数据训练时，为了防止过拟合，添加Dropout层，设置一个0~1之间的小数。



预训练阶段 retrain_batch_size 不容易传参，报错TypeError: Cannot interpret feed_dict key as Tensor: Can not convert a int into a Tensor. 所以目前手动在generator.py中修改

=》 如果要改变windows size和batch size要在config中同时修改clstm和generator配置



window size增大为20->60，batch size减小为800->512 =》 召回率提升，f1提升

```
step: 6200, loss: 0.215567, accuracy: 0.930756, precision: 0.893921, recall: 0.295276, f1: 0.443918
```



改大滑窗的好处：

1. 滑窗应适当大于异常点群的跨度，查看plot数据集可知，这个跨度一般要大于20
2. 改善样本不均衡问题 95:5  —> 90:10

训练结果确实有所改善





Retrain_epoch_num增加，训练效果改善，可能之前没有完全收敛

参数：

```
batch_size: 512
num_batches: 124
epochs: 1000
steps: 124*1000 = 124000
```

训练效果：

```
Test
2019-12-15T16:23:39.217199: step: 124000, loss: 0.148528, accuracy: 0.952683, precision: 0.90994, recall: 0.542778, f1: 0.67996
```





fm 方法用到了discriminator中间层的输出



**问题：**

加入GAN后效果没有单纯C-LSTM好





下周：

1. 调节超参数
2. 可以把训练过程用tensorboard plot出来，可能会对调参有帮助