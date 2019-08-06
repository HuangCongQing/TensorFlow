
>了解如何在 TensorFlow 中创建和修改张量。
了解 Pandas 的基础知识。
使用 TensorFlow 的一种高级 API 开发线性回归代码。
尝试不同的学习速率。 

快速了解 tf.estimator API     ------使用 tf.estimator 会大大减少代码行数。
```
import tensorflow as tf

# Set up a linear classifier.
classifier = tf.estimator.LinearClassifier()

# Train the model on some example data.
classifier.train(input_fn=train_input_fn, steps=2000)

# Use it to predict.
predictions = classifier.predict(input_fn=predict_input_fn)
```

### Tensorflow工具包
```
Estimator (tf.estimator)	高级 OOP API。
tf.layers/tf.losses/tf.metrics	用于常见模型组件的库。
TensorFlow	低级 API
```

### 常用超参数

* steps：训练迭代的总次数。一步计算一批样本产生的损失，然后使用该值修改一次模型的权重。
* batch size：单步的样本数量（随机选择）。例如，SGD 的批次大小为 1。

periods：控制报告的粒度。例如，如果 periods 设为 7 且 steps 设为 70，则练习将每 10 步输出一次损失值（即 7 次）。与超参数不同，我们不希望您修改 periods 的值。请注意，修改 periods 不会更改模型所学习的规律。