# TSI+CNN训练及使用手册

## Compile Model

* **optimizer**: 可以是字符串形式给出的优化器名字，也可以是函数形式，使用函数形式可以设置学习率、动量和超参数

  - [ ] “**sgd**”  或者

    ```python
     tf.optimizers.SGD(lr = 学习率, decay = 学习率衰减率, momentum = 动量参数）
    ```

  - [ ]  “**adagrad**" 或者 

    ```python
    tf.keras.optimizers.Adagrad(lr = 学习率,  decay = 学习率衰减率）
    ```

  - [ ]  ”**adadelta**" 或者 

    ```python
    tf.keras.optimizers.Adadelta(lr = 学习率, decay = 学习率衰减率）
    ```

  - [x] “**adam**" 或者

    ```python
     tf.keras.optimizers.Adam(lr = 学习率, decay = 学习率衰减率）
    ```

  - [ ] others(maybe you need Scientific Internet Access)

    <img src="img/optimizer.png" alt="optimizer" style="zoom:50%;" />

    [tf.keras.optimizers]: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers

* **loss**: 可以是字符串形式给出的损失函数的名字，也可以是函数形式

  ​		⚠️可能是版本缘故，必须使用损失函数的名字（下面加粗的部分），而不能直接调用函数，否则会报错

  - [ ] ”**mse**" 或者

    ```python
     tf.keras.losses.MeanSquaredError()
    ```

  - [ ]  "**sparse_categorical_crossentropy**" 或者 

    ```python
    tf.keras.losses.SparseCatagoricalCrossentropy(from_logits = False)
    ```

  - [x] "**binary_cross_entropy**"或者

    ```python
    tf.keras.losses.BinaryCrossentropy()
    ```

  - [ ] others(maybe you need Scientific Internet Access)

    <img src="img/loss.png" alt="loss" style="zoom:50%;" />

    [tf.keras.losses]: https://www.tensorflow.org/api_docs/python/tf/keras/losses

* **metrics**: 标注网络评价指标

  - [x] "**accuracy**" : y_ 和 y 都是数值，如y_ = [1] y = [1] #y_为真实值，y为预测值_
  
  - [ ] “**sparse_accuracy**":y_和y都是以独热码 和概率分布表示，如y_ = [0, 1, 0], y = [0.256, 0.695, 0.048]
  
  - [ ] "**sparse_categorical_accuracy**" :y_是以数值形式给出，y是以 独热码给出，如y_ = [1], y = [0.256 0.695, 0.048]
  
  - [ ] others
  
    <img src="img/metrics.png" alt="metrics" style="zoom:50%;" />
  
    [tf.keras.optimizers]: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
  
    
## EarlyStopping  

* 作用：当受监视的指标停止改进时停止培训

* 本项目中使用的

  ```python
  es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
  ```

* ```python
  tf.keras.callbacks.EarlyStopping(
      monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
      baseline=None, restore_best_weights=False
  )
  ```

  | `monitor`              | Quantity to be monitored.                                    |
  | ---------------------- | ------------------------------------------------------------ |
  | `min_delta`            | Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change of less than min_delta, will count as no improvement. |
  | `patience`             | Number of epochs with no improvement after which training will be stopped. |
  | `verbose`              | verbosity mode.                                              |
  | `mode`                 | One of `{"auto", "min", "max"}`. In `min` mode, training will stop when the quantity monitored has stopped decreasing; in `"max"` mode it will stop when the quantity monitored has stopped increasing; in `"auto"` mode, the direction is automatically inferred from the name of the monitored quantity. |
  | `baseline`             | Baseline value for the monitored quantity. Training will stop if the model doesn't show improvement over the baseline. |
  | `restore_best_weights` | Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used. |

## Fit/Train

```python
model.fit(x=train_ds,epochs=100,callbacks=cb, steps_per_epoch=20)
```

* x: 训练集的输入数据
* epochs: 迭代次数（不一定越高越好，太多次可能会出现过拟合现象，10次就可以到达85%以上的准确率了，可以适当增加；**如果发生过拟合问题，建议换回最初的模型重新跑**）
* callbacks: 回调函数（本模型中使用的是上面讲的ES）
* steps_per_epoch: 每次迭代中的步数
  * ` batchSize * steps_per_epoch = 每个epoch中用到的数据条数`



## 路径问题

尽量使用相对路径，请自行python中查阅相对路径的写法

⚠️需要特别特别注意的是，`相对路径`相对的是`开始运行的那个文件的路径`，而不是`使用相对路径的文件所在路径`