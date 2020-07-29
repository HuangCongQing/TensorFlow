# TensorFlow
TensorFlow学习的一些练习代码

本文tf1.x版本：'1.14.0' py35

* 个人语雀笔记：https://www.yuque.com/huangzhongqing/tensorflow

* jupyter运行：在本目录终端运行`jupyter-notebook`命令来打开http://localhost:8888/tree

* 官网：https://www.tensorflow.org/
    * github： https://github.com/tensorflow
    * tensorflow2.0 beta教程 :https://www.tensorflow.org/beta/
* 查看jupyternotebook：https://nbviewer.jupyter.org/
* 网络编辑和运行NotebooksGoogle Colab：https://colab.research.google.com/
* tensors_flowing
![tensors_flowing图](https://www.tensorflow.org/images/tensors_flowing.gif)



简单粗暴 TensorFlow 2 文档：https://tf.wiki/zh/basic.html


### Tutorial(教程)

* [安装](00TensorFlow基本概念和安装方法.ipynb)
* [查看tf和py版本](02.01tensorflow学习笔记一：查看版本号.ipynb)

#### [bilibili-深度学习框架Tensorflow学习与应用](https://www.bilibili.com/video/BV1kW411W7pZ)[初学2020.07][tf1.x][13.5h]

* 环境：tf1.x：'1.14.0' py35
* 莫烦没有讲具体函数的功能，相比还是这个视频好
* 笔记：https://www.yuque.com/huangzhongqing/tensorflow/yiggki

code:[深度学习框架Tensorflow学习与应用](./深度学习框架Tensorflow学习与应用)
* [x] 第一课 Tensorflow简介，Anaconda安装，Tensorflow的CPU版本安装。
* [x] **第二课 Tensorflow的基础使用，包括对图(graphs),会话(session),张量(tensor),变量(Variable)的一些解释和操作。**
* [ ] 第三课 Tensorflow线性回归以及分类的简单使用，softmax介绍。
* [ ] 第四课 交叉熵(cross-entropy)，过拟合，dropout以及Tensorflow中各种优化器的介绍。
* [ ] 第五课 使用Tensorboard进行结构可视化，以及网络运算过程可视化。
* [ ] 第六课 卷积神经网络CNN的讲解，以及用CNN解决MNIST分类问题。
* [ ] 第七课 递归神经网络LSTM的讲解，以及LSTM网络的使用。
* [ ] 第八课 保存和载入模型，使用Google的图像识别网络inception-v3进行图像识别。
* [ ] 第九课 Tensorflow的GPU版本安装。设计自己的网络模型，并训练自己的网络模型进行图像识别。
* [ ] 第十课 多任务学习以及验证码识别。
* [ ] 第十一课 word2vec讲解和使用，cnn解决文本分类问题。
* [ ] 第十二课 语音处理以及使用LSTM构建语音分类模型。

部分参考： https://github.com/Parker-Lyu/TensorFLow-Learning

---


* [视频·google基于tensorflow的机器学习课程(免费中文)](https://developers.google.cn/machine-learning/crash-course/)[学於20190805]


* [bilibili视频·莫凡-tensorflow搭建自己的神经网络](https://www.bilibili.com/video/av16001891?zw)[学於201710]
>视频发布很早，但对应讲解莫凡大佬[github代码](https://github.com/MorvanZhou/Tensorflow-Tutorial)有更新，所以，看视频的时候，代码还是看github上的吧



#### [视频·google基于tensorflow的机器学习课程(免费中文)](https://developers.google.cn/machine-learning/crash-course/)

* 机器学习术语表：https://developers.google.cn/machine-learning/glossary/
* [01框架处理（15 分钟）机器学习中的监督学习-标签&特征·样本&模型·回归&分类](./google_tensorflow/01框架处理.md)

* [02深入了解机器学习（20 分钟）什么是损失函数，权重weight和 bias 是什么](/google_tensorflow/02深入了解机器学习.md)
* [03降低损失（60 分钟）两种梯度下降，SGD,及对lr学习率的实验](/google_tensorflow/03降低损失.md)
* [04使用 TensorFlow 基本步骤（60 分钟）不能不懂的 TensorFlow](/google_tensorflow/04使用TensorFlow的基本步骤.md)
* [05泛化（15 分钟）什么是过拟合，怎样评价一个模型的好坏，把数据集分成测试和训练两部分](/google_tensorflow/泛化.md)
* 06训练及测试集（25 分钟）验证把数据集分成两部分的好处
* 07验证（40 分钟）担心过拟合？在测试和训练集外多弄一个验证集
* 08表示法（65 分钟）特征工程，75% 机器学习工程师的时间都在干的事
* 09特征组合（70 分钟）明白什么是特征组合，怎么用 TensorFlow 实现
* 10正则化：简单性（40 分钟）L2 正则化，学习复杂化和普遍化的取舍
* 11逻辑回归（20 分钟）理解逻辑回归，探索损失函数和正则化
* 12分类（90 分钟）评估一个逻辑回归模型的正确性和精度
* 13正则化：稀松性（45 分钟）L2 的其他种类
* 14介绍神经网络（40 分钟）隐藏层，激活函数
* 15训练神经网络（40 分钟）反向传播
* 16多种类神经网络（50 分钟）理解多类分类器问题，Softmax，在 TensorFlow 中实现 Softmax 结果。
* 17嵌入（80 分钟）什么是嵌入，这是干什么的，怎样用好。


#### [bilibili视频·莫凡-tensorflow搭建自己的神经网络](https://www.bilibili.com/video/av16001891?zw)[初学2019.08]



#### [bilibili视频·tensorflow2.0入门与实战 2019年](https://www.bilibili.com/video/av62215565?from=search&seid=1287497745063342076)[初学2019.08]

* tensorflow2.0教程文档：https://github.com/czy36mengfei/tensorflow2_tutorials_chinese
* tensorflow2.0 beta :https://www.tensorflow.org/beta/
    * 第一个巨大差异：移除tf.get_variable, tf.variable_scope, tf.layers，强制转型到基于Keras的方法，也就是用tf.keras。
    * 训练方面：使用keras和eager execution（动态图机制）(eager notebook开发更好)
    * 服务器，边缘设备，网页，any语言0rp平台，皆可训练部署模型
    * tf.keras,Eager模式和自定义训练说,tf.data，tf.function(自动图运算)。模型保存于可视化，Tensorboard可视化



### install

* 安装CPU版本
`pip install tensorflow==2.0.0-alpha0`

2019年7月新发布beta0
`pip install tensorflow==2.0.0-beta0`

推荐：用豆瓣源安装多个python包(含tensorflow2.0

`pip install numpy pandas matplotlib sklearn tensorflow==2.0.0-alpha0 -i https://pypi.doubanio.com/simple/`

* 安装GPU版本（CUDA，cuDNN已安装）

`pip install numpy pandas matplotlib sklearn tensorflow==2.0.0-alpha0 -i https://pypi.doubanio.com/simple/`


* [03机器学习原理-线性回归](./tensorflow2.0/)











### contents(目录)


### Resource(资源)


