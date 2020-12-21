# 邱博模型

## 整体框架

### 数据集

* aiopsdata
* 单变量

![屏幕截图 2020-11-25 123247](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_qj_model\屏幕截图 2020-11-25 123247.png)

### 模型的构建和训练

#### **ConvLSTM**

* LSTM是一种特殊的RNN（有两个传输状态）

  ![v2-e4f9851cad426dfe4ab1c76209546827_r](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_qj_model\v2-e4f9851cad426dfe4ab1c76209546827_r.jpg)

* ConvLSTM就是在LSTM之前加卷积操作，邱博的模型架构为三层卷积池化+LSTM+softmax
* 训练时，训练数据以窗口的形式传到模型里进行训练

## 运行结果截图

![屏幕截图 2020-11-25 130719](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_qj_model\屏幕截图 2020-11-25 130719.png)