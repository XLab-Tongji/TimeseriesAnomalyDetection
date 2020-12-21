#  RNN简述

* **结构**：RNN中的每个节点都有关联，如下图所示，Xt表示t时刻的输入，Ot是t时刻对应的输出， St是t时刻的存储记忆。对于RNN中的每个单元，输入分为两个部分：1）当前时刻的真正的输入Xt；2）前一时刻的存储记忆St-1。

  ![20190510141142272](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_rnn_model\20190510141142272.png)

* **常见运用**：RNN 常用于序列是相互依赖的（有限或无限）数据流，所以适合时间序列的数据，它的输出可以是一个序列值或者一序列的值。

# RNN用于时间序列异常检测模型详解

## 整体框架

### 数据集

六种数据集可自由选择

#### ecg(双变量)

![屏幕截图 2020-11-25 121138](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_rnn_model\屏幕截图 2020-11-25 121138.png)

#### gesture(双变量)

![屏幕截图 2020-11-25 121443](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_rnn_model\屏幕截图 2020-11-25 121443.png)

#### nyc_taxi(三变量)

![屏幕截图 2020-11-25 121629](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_rnn_model\屏幕截图 2020-11-25 121629.png)

#### power_demand(单变量)

![屏幕截图 2020-11-25 121802](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_rnn_model\屏幕截图 2020-11-25 121802.png)

#### respiration(单变量)

![屏幕截图 2020-11-25 121901](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_rnn_model\屏幕截图 2020-11-25 121901.png)

#### space_shuttle(单变量)

![屏幕截图 2020-11-25 121941](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_rnn_model\屏幕截图 2020-11-25 121941.png)

### RNN模型的构建和训练(使用ecg数据)

![屏幕截图 2020-11-25 122135](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_rnn_model\屏幕截图 2020-11-25 122135.png)


## 运行结果截图

![屏幕截图 2020-11-25 123957](D:\course\3up\software engineering\lab work\week2_CNN_time_series_classifier\cai_rnn_model\屏幕截图 2020-11-25 123957.png)

## 框架意义

提供了数据集的自由选择，后续能加入模型的自由选择，可作为项目的整体框架。