
<map>
  <node ID="root" TEXT="基于时间序列的深度异常检测">
    <node TEXT="项目过程" ID="DOqQXH4HCV" STYLE="bubble" POSITION="right">
      <node TEXT="准备阶段" ID="GTNg8NUj7p" STYLE="fork">
        <node TEXT="阅读文献" ID="ThaKzqJa7f" STYLE="fork">
          <node TEXT="异常检测领域综述性论文（Kerr）" ID="qdtvLetslP" STYLE="fork">
            <node TEXT="DEEP LEARNING FOR ANOMALY DETECTION: A SURVEY" ID="dUghAVuDOG" STYLE="fork">
              <node TEXT="关键字解读" ID="pf0c2q8y8P" STYLE="fork">
                <node TEXT="异常检测（也称为异常值检测）" ID="HH1iOJlZ5T" STYLE="fork">
                  <node TEXT="目标是以数据驱动的方式确定所有与众不同的异常数据。" ID="nE088mQYHr" STYLE="fork"/>
                  <node TEXT="异常数据和奇异值的区别" ID="G8Ug1Swefh" STYLE="fork"/>
                </node>
                <node TEXT="深度学习" ID="ErUSd6npJj" STYLE="fork">
                  <node TEXT="机器学习的子集" ID="uzqEIDMswo" STYLE="fork">
                    <node TEXT="优点" ID="RNV1UI3el8" STYLE="fork">
                      <node TEXT="性能好" ID="VhaSp0DFdf" STYLE="fork"/>
                      <node TEXT="灵活性较高" ID="VVpGqrm5JW" STYLE="fork"/>
                      <node TEXT="数据规模增加时，深度学习的效率高于传统的机器学习" ID="rBaTR7jK9b" STYLE="fork"/>
                    </node>
                  </node>
                </node>
              </node>
              <node TEXT="深度异常检测的动机与挑战" ID="t9hXwDWX8h" STYLE="fork">
                <node TEXT="传统方法的性能不够好，因为它无法捕获数据中的复杂结构" ID="4qCX0Gj6P2" STYLE="fork"/>
                <node TEXT="传统方法的可扩展性较差，而数据集却是越来越大" ID="GrtSWm9jaI" STYLE="fork"/>
                <node TEXT="传统方法的适应性在减弱，因为正常行为和异常行为的边界越来越模糊，而且还在不断发展" ID="qQMm9gtfgK" STYLE="fork"/>
                <node TEXT="传统方法对手动特征工程依赖性较高，DAD具有自动特征学习功能" ID="oUJQ6sKbWr" STYLE="fork"/>
              </node>
              <node TEXT="分类" ID="qr1mpxVsdP" STYLE="fork">
                <node TEXT="按输入数据" ID="8Aqrj5BjhS" STYLE="fork">
                  <node TEXT="序列" ID="CuO4EOv0le" STYLE="fork">
                    <node TEXT="音频序列" ID="Ey2PV2o0pU" STYLE="fork"/>
                    <node TEXT="蛋白质序列" ID="JUW0MGACEz" STYLE="fork"/>
                    <node TEXT="时间序列" ID="d7kDRW6qXV" STYLE="fork"/>
                  </node>
                  <node TEXT="非序列" ID="zwKurESosR" STYLE="fork">
                    <node TEXT="图像" ID="KdhNJ2zL8M" STYLE="fork"/>
                  </node>
                </node>
                <node TEXT="按是否监督" ID="QTYhgDVF8T" STYLE="fork">
                  <node TEXT="监督式" ID="pF1SVE6nSl" STYLE="fork">
                    <node TEXT="加标签" ID="dwdN3EpAEP" STYLE="fork"/>
                  </node>
                  <node TEXT="半监督式" ID="7nmiwWeHxi" STYLE="fork">
                    <node TEXT="仅单标签" ID="wXnjHUqPzR" STYLE="fork"/>
                  </node>
                  <node TEXT="无监督" ID="IYMDxBDKyY" STYLE="fork">
                    <node TEXT="无标签" ID="W0MvLCVjRe" STYLE="fork"/>
                  </node>
                </node>
                <node TEXT="按异常的分类" ID="IQVGFgTrEl" STYLE="fork">
                  <node TEXT="点异常" ID="PteZTSkChe" STYLE="fork"/>
                  <node TEXT="集体异常" ID="6dWkSl7r9Z" STYLE="fork">
                    <node TEXT="单个出现没有问题，成群结队出现就可能是异常" ID="LWW5lTGcgW" STYLE="fork"/>
                  </node>
                  <node TEXT="上下文异常" ID="Bg4s2VTOMr" STYLE="fork"/>
                </node>
                <node TEXT="按输出分类" ID="3qhd5xYZZO" STYLE="fork">
                  <node TEXT="异常分值" ID="WAqPnUTKh8" STYLE="fork">
                    <node TEXT="给出一个异常程度，然后域专家根据经验设置阈值来判定是否是异常" ID="wVNbln7uNi" STYLE="fork"/>
                  </node>
                  <node TEXT="标签" ID="cFXPFt3WR8" STYLE="fork"/>
                </node>
              </node>
              <node TEXT="应用" ID="VmLcOnmdeh" STYLE="fork">
                <node TEXT="入侵检测" ID="D72mcMNlC0" STYLE="fork">
                  <node TEXT="网络" ID="YOo1vc608c" STYLE="fork"/>
                  <node TEXT="主机" ID="RC5PSslDKI" STYLE="fork"/>
                </node>
                <node TEXT="欺诈检测" ID="vF2spVh0Zx" STYLE="fork">
                  <node TEXT="银行" ID="XoETgVfhCs" STYLE="fork"/>
                  <node TEXT="网络" ID="hm8e8ng1bU" STYLE="fork"/>
                </node>
                <node TEXT="恶意软件检测" ID="qquex3xaOl" STYLE="fork"/>
                <node TEXT="医疗异常检测" ID="lU2Y59uwkN" STYLE="fork"/>
                <node TEXT="社交异常检测" ID="mORhsjSwNo" STYLE="fork">
                  <node TEXT="垃圾邮件" ID="JNCDzhDm1V" STYLE="fork"/>
                  <node TEXT="网络骗子" ID="0MQijJVbdQ" STYLE="fork"/>
                  <node TEXT="虚假用户" ID="zyEcNsqRfe" STYLE="fork"/>
                  <node TEXT="谣言散布" ID="rC2mXb5LLA" STYLE="fork"/>
                  <node TEXT="性侵犯" ID="g5NTResNL3" STYLE="fork"/>
                </node>
                <node TEXT="日志异常检测" ID="ATJcOwAk3G" STYLE="fork"/>
                <node TEXT="物联网数据异常检测" ID="c8xYmjaf32" STYLE="fork"/>
                <node TEXT="时间序列异常检测" ID="YeMxj353UK" STYLE="fork"/>
              </node>
              <node TEXT="现有模型" ID="06aUzOEONA" STYLE="fork">
                <node TEXT="时空网络模型（STN）" ID="Xtk4eyOq3w" STYLE="fork">
                  <node TEXT="一般的深度学习只能分析时间或空间的特征" ID="DC7nO9jT3r" STYLE="fork"/>
                  <node TEXT="CNN分析空间特征" ID="bV9Q3JsEDi" STYLE="fork"/>
                  <node TEXT="LSTM分析时间特征" ID="VrSHYRVF9b" STYLE="fork"/>
                </node>
                <node TEXT="总和产品网络（SPN）" ID="vwaAbaGT2k" STYLE="fork"/>
                <node TEXT="词向量模型（Word2vec）" ID="RAs8GaNJpQ" STYLE="fork"/>
                <node TEXT="生成模型" ID="84JzSGZ5eg" STYLE="fork">
                  <node TEXT="变异自编器（VAE）" ID="OW8g4qKHUl" STYLE="fork"/>
                  <node TEXT="生成对抗网络（GAN）" ID="xNgLZv9Nv0" STYLE="fork"/>
                </node>
                <node TEXT="卷积神经网络（CNN）" ID="8Z7VTtA92G" STYLE="fork"/>
                <node TEXT="序列模型" ID="iqpPJXTf2P" STYLE="fork">
                  <node TEXT="RNN循环神经网络" ID="k0JuEGwAFi" STYLE="fork"/>
                </node>
              </node>
            </node>
          </node>
          <node TEXT="针对时间序列异常检测的研究性论文" ID="rAGdbo7fdC" STYLE="fork">
            <node TEXT="Profile" ID="TZFDlAr5Bj" STYLE="fork">
              <node TEXT="Matrix Profile I: All Pairs Similarity Joins for Time Series:A Unifying View that Includes Motifs, Discords and Shapelets（晓楠）" ID="IEkl06IrkP" STYLE="fork">
                <node TEXT="Matrix Profile" ID="Sqt4vD8W6m" STYLE="fork">
                  <node TEXT="由UCR（加州⼤学河滨分校）提出的⼀个时间序列的分析算法。通过⼀个时间序列，可计算出它的MP，表示的是子序列之间的潜在关系（各种距离），MP也是⼀个向量（时间序列）。" ID="Y938DLjoiu" STYLE="fork"/>
                  <node TEXT="Shapelets" ID="T5MTfl9Rna" STYLE="fork">
                    <node TEXT="代表⼀类可以在分类场景中提供直接的解释性和⻅解的时间序列⼦序列[6]，并且基于Shapelet的模型在各种研究中都被证明是有前景的" ID="XNJ9sGQhYP" STYLE="fork"/>
                    <node TEXT="太难了，相关资料并不多，学姐研究了几周后放弃了这个方向转向和我们一起研究TSI" ID="xxv93aq7ut" STYLE="fork"/>
                  </node>
                </node>
              </node>
            </node>
            <node TEXT="Time Series to Image" ID="dlK2LITYGa" STYLE="fork">
              <node TEXT="TSI: Time series to imaging based model for detecting anomalous energy consumption in smart buildings（Kerr）" ID="lU1FDUH5ax" STYLE="fork">
                <node TEXT="背景" ID="0Gt2EGkXiE" STYLE="fork"/>
                <node TEXT="现状" ID="gcjnG2fZ9s" STYLE="fork">
                  <node TEXT="⽬前处理这种⼤规模数据的⽅式还是需要域专家的经验，特征提取" ID="lTgS26aUOG" STYLE="fork">
                    <node TEXT="领域知识基础" ID="YZUmYXoERD" STYLE="fork"/>
                    <node TEXT="⼿动" ID="qrUHqaRKQ9" STYLE="fork"/>
                  </node>
                  <node TEXT="出现了一些新方法" ID="DgjMuhE9ds" STYLE="fork">
                    <node TEXT="将时间序列转化为图像" ID="tgxKSCmRmu" STYLE="fork">
                      <node TEXT="无须领域专家参与" ID="2Ffva7pEtP" STYLE="fork"/>
                      <node TEXT="自动提取特征" ID="sf4TQm7GpD" STYLE="fork"/>
                    </node>
                    <node TEXT="对图像进行学习分类" ID="9GNS9A4sDH" STYLE="fork">
                      <node TEXT="存储小" ID="4nzEGgJeIb" STYLE="fork"/>
                      <node TEXT="分析速度快" ID="AvhJ1Y0SaR" STYLE="fork"/>
                    </node>
                  </node>
                </node>
                <node TEXT="模型" ID="ty915hTYGU" STYLE="fork"/>
                <node TEXT="数据集" ID="4ysxZcSDoo" STYLE="fork">
                  <node TEXT="REFIT 电⽓负载测量数据集" ID="9b7M1DlqNF" STYLE="fork">
                    <node TEXT="英国的20幢房子" ID="KZYv0gvrCr" STYLE="fork"/>
                    <node TEXT="2年" ID="joNfR55ujk" STYLE="fork"/>
                    <node TEXT="无标签" ID="emGRQTbUop" STYLE="fork"/>
                    <node TEXT="每个房子里面的人都照常工作生活" ID="phSHnlBURY" STYLE="fork"/>
                  </node>
                </node>
                <node TEXT="测试及结果" ID="gVl2QOYT94" STYLE="fork">
                  <node TEXT="TSI model 在3个不同房⼦的数据上的测试结果" ID="qqI3B6ixDn" STYLE="fork">
                    <node TEXT="house 1" ID="4MLrFaawYg" STYLE="fork"/>
                    <node TEXT="house 2" ID="wzyAUbUwKV" STYLE="fork"/>
                    <node TEXT="house 3" ID="FI4xjXZM5F" STYLE="fork"/>
                  </node>
                  <node TEXT="测试没有imaging这⼀步效果如何" ID="4diGEgjEBc" STYLE="fork">
                    <node TEXT="house 1" ID="YIGvdXhgZF" STYLE="fork"/>
                    <node TEXT="house 2" ID="tBCoWsXcPE" STYLE="fork"/>
                    <node TEXT="house 3" ID="3BQpw4HcFJ" STYLE="fork"/>
                    <node TEXT="证明了image的重要性，这个结果可以看出没有image模型⼏乎检测不到异常" ID="ZhgOWqENsd" STYLE="fork"/>
                  </node>
                  <node TEXT="对比主成分分析" ID="dvGScdIcDN" STYLE="fork">
                    <node TEXT="house 1" ID="ajpoJGJhSc" STYLE="fork"/>
                    <node TEXT="house 2" ID="f0rVVWy7j4" STYLE="fork"/>
                    <node TEXT="house 3" ID="GUne1X2PNM" STYLE="fork"/>
                    <node TEXT="给TSI模型的准确性做个参照，可以看到主成分分析的结果准确率只有50%左右" ID="fJDvs83XpL" STYLE="fork"/>
                  </node>
                </node>
              </node>
            </node>
            <node TEXT="Others" ID="2MtIJ08xSi" STYLE="fork">
              <node TEXT="A Novel Technique for Long-Term Anomaly Detection in the Cloud（2014 引用153） (yuan)" ID="Y0XvTFEbjZ" STYLE="fork">
                <node TEXT="引言" ID="v2DkcMULCZ" STYLE="fork">
                  <node TEXT="云计算随着发展在现代社会中占据着越来越重要的地位，文章的作者是twitter的一名工程师，他提到twitter当下面临的一个问题是如何自动检测云平台上的长期异常，即long-term anomalies。" ID="u1F3XBG7y0" STYLE="fork">
                    <node TEXT="注：长期过程中的异常之所以难以检测，是因为时间序列有潜在的占主导性的趋势(trend)，若不考虑trend采用短期异常检测的做法，误报的可能性很大，因为没有考虑到数据本身的趋势。" ID="samihD7Igx" STYLE="fork"/>
                  </node>
                  <node TEXT="为此作者在ESD(generalized Extreme Studentized Deviate)的基础上建立了一种新的统计学方法，称之为分段中值法(picewise median)，并在实际生产数据上测试了其性能。" ID="SEqP1MTmGo" STYLE="fork"/>
                  <node TEXT="ESD：检测单变量异常值的一种统计学方法，数学表达式如下：" ID="QSQg4MeJ3y" STYLE="fork"/>
                </node>
                <node TEXT="公式基础" ID="5IAZhEZUN9" STYLE="fork">
                  <node TEXT="时间序列X是x_t的集合，其中t=0,1,2..." ID="RXMgze0Gwj" STYLE="fork"/>
                  <node TEXT="R_X=X-S_X-T_X" ID="C5QMNvxC6f" STYLE="fork"/>
                  <node TEXT="解释：X可分解为三部分" ID="TxjDmMUbVC" STYLE="fork">
                    <node TEXT=" Sx是周期性成分，描述序列的周期性变化" ID="wCHzVyPmKS" STYLE="fork"/>
                    <node TEXT="Tx是趋势成分，描述序列的总体趋势（非周期性）" ID="f4yvcCxEsH" STYLE="fork"/>
                    <node TEXT="Rx是X去掉Sx和Tx后的剩余成分" ID="boQdpscSPQ" STYLE="fork"/>
                  </node>
                  <node TEXT="异常检测主要针对Rx进行检测" ID="CAOdURWCI9" STYLE="fork"/>
                </node>
                <node TEXT="新技术：分段中值法" ID="dW3wjXyRqd" STYLE="fork">
                  <node TEXT="周期性成分比较容易确定，趋势成分(trend)如何提取将直接影响结果的好坏，所以如何确定trend十分关键。" ID="lzfMv6u7um" STYLE="fork"/>
                  <node TEXT="经过实际实验发现，以中值代替得到的trend从X中减去，这种方法的性能好于直接减去Tx，能减少误报的产生，如下图：" ID="2vIsSk5982" STYLE="fork"/>
                  <node TEXT="但是这种方法只对trend比较平的序列比较有效，在有显著trend的长期序列中表现很差。" ID="jgmnnhIvmh" STYLE="fork"/>
                  <node TEXT="作者考察了现有的两种提取trend的方法：STL trend和分位数回归法(quantile regression)，然后提出了新技术——分段中值法，并比较了它们的性能。" ID="q13pEJ6Ey2" STYLE="fork"/>
                  <node TEXT="STL trend 概要" ID="B86VTUzciE" STYLE="fork">
                    <node TEXT="从序列中减去估计的trend，将得到的新序列划分为子周期序列(sub-cycle series)" ID="bWMfmHYNQI" STYLE="fork"/>
                    <node TEXT="对每个子周期序列运用LOESS(局部加权回归)，估计Sx，从原始序列中减去Sx得到新的序列，对新序列运用LOESS得到新的trend估计值" ID="EJ3fziETY6" STYLE="fork"/>
                    <node TEXT="重复上述步骤直到新的trend估计值不再改变或改变得很小" ID="pvYRXHEtlK" STYLE="fork"/>
                    <node TEXT="注：局部加权回归LOESS" ID="L1lHjOgaKp" STYLE="fork"/>
                    <node TEXT="局部加权回归：以一个点x为中心，向前后截取一段长度为frac的数据，对于该段数据用权值函数w做一个加权的线性回归。对于所有的n个数据点则可以做出n条加权回归线，每条回归线的中心值的连线则为这段数据的Lowess曲线。" ID="2e0ClJ3JcN" STYLE="fork"/>
                  </node>
                  <node TEXT="分位数回归法" ID="K07j0Ut2bO" STYLE="fork">
                    <node TEXT="分位数回归是把分位数的概念融入到普通的回归里，所谓的0.9分位数回归，就是希望回归曲线之下能够包含90%的数据点，这也是分位数的概念。" ID="FleEn9FQaC" STYLE="fork"/>
                    <node TEXT="用B-Spline曲线做分位数回归已被证实是很有效的提取trend的方法，但是这种方法适用于两周以上的长序列，当时间小于两周时会过拟合，这就意味着假如出现大块的异常数据隔断了正常数据，这种方法的性能会受到严重影响。" ID="Uq2XTstTKt" STYLE="fork"/>
                  </node>
                  <node TEXT="分段中值法" ID="fgRUJBrqLJ" STYLE="fork">
                    <node TEXT="分段中值法的思想是用分段取中值的方法对序列的trend做近似，算法如下图" ID="eoo0KckwCE" STYLE="fork"/>
                    <node TEXT="窗口大小的选取要注意，一般包括两个完整周期，至少要有一个周期" ID="G821j4YHV8" STYLE="fork"/>
                    <node TEXT="0.49是多次实验得到的序列能容纳的最大异常率" ID="0LPy6yZAVk" STYLE="fork"/>
                  </node>
                </node>
                <node TEXT="上述三种方法的比较" ID="ktli8A3b3u" STYLE="fork">
                  <node TEXT="三种方法提取trend的示意图如下" ID="RaQIOXPjjf" STYLE="fork"/>
                  <node TEXT="有效性比较" ID="gN0SoDy7ib" STYLE="fork">
                    <node TEXT="可以看出三种方法都能检测出比较明显的异常，但相比于piece median，另外两种方法的误报更多。STL是将很多正常数据检测为了异常数据，quantile b-spline是没有检测出很多本该是异常的数据。" ID="lk8Gy9DlzM" STYLE="fork"/>
                  </node>
                  <node TEXT="实时性能比较" ID="CP1ztWtuUv" STYLE="fork">
                    <node TEXT="当分析三个月的以分(minute)为单位的序列数据时，piecewise median只用了四分多钟，随着数据量的增大，提升的效果是显著的。" ID="h9NucZAhIo" STYLE="fork"/>
                  </node>
                  <node TEXT="注入分析" ID="y8s9F5oxND" STYLE="fork">
                    <node TEXT="实际中很难采集到异常数据，可以采用异常注入的方法进行测试" ID="s4rIaBZgrj" STYLE="fork">
                      <node TEXT="异常注入在两个维度上进行：注入时间和异常量级" ID="58aOWj0WxE" STYLE="fork"/>
                      <node TEXT="三种方法异常注入实验结果如下图" ID="T3176kFP1K" STYLE="fork"/>
                    </node>
                  </node>
                </node>
              </node>
              <node TEXT="A Deep Neural Network for Unsupervised Anomaly Detection and Diagnosis in Multivariate Time Series Data（Brian）" ID="PEOvWmjX0N" STYLE="fork">
                <node TEXT="2019年 AAAI会议发表" ID="LnXcMklQY9" STYLE="fork"/>
                <node TEXT="本论文主要解决了多变量时间序列数据中的异常检测中的几个问题：" ID="8oVdgWStoy" STYLE="fork">
                  <node TEXT="时间依赖性问题，能捕获在不同时间步长中的时间依赖关系" ID="kDVmBbaO9O" STYLE="fork"/>
                  <node TEXT="实际应用中的噪声对模型预测结果影响较大，此模型尽可能将噪声的影响降到最低" ID="go85rZrFCg" STYLE="fork"/>
                  <node TEXT="在现实应用中，可以根据异常事件的严重程度，在异常检测的基础上进行异常评分，分析出现异常的根本原因" ID="SvdBjQ4Pqd" STYLE="fork"/>
                </node>
                <node TEXT="本论文的主要贡献" ID="wW2J0gXArB" STYLE="fork">
                  <node TEXT="提出了一种多尺度卷积循环编解码器（MSCRED），来进行多变量时间序列数据的异常检测和诊断，较好地解决了上述三个问题" ID="wjCmuvtdV7" STYLE="fork"/>
                  <node TEXT="签名矩阵" ID="zXlusxO8fb" STYLE="fork">
                    <node TEXT="描述不同时间步长中系统状态的多个级别，可以表示不同变量的时间序列之间的相互关系，级别就是用来表示异常事件的严重程度" ID="BL64bHydYh" STYLE="fork"/>
                  </node>
                  <node TEXT="" ID="aFk4iSvvu0" STYLE="fork">
                    <node TEXT="使用卷积编码器对时间序列编码" ID="Vjy7VM94XW" STYLE="fork"/>
                    <node TEXT="并开发了基于注意力的ConvLSTM网络来捕获时间模式ConvLSTM被开发用于捕获视频序列中的时间信息，但其性能可能会随着序列长度的增加而恶化。" ID="lA1LwAgAoy" STYLE="fork"/>
                    <node TEXT="加入了注意力机制（Attention Based ConvLSTM）后，可以跨不同的时间步自适应地选择相关的隐藏状态。" ID="VWDeflhw4X" STYLE="fork"/>
                    <node TEXT="利用卷积译码器重构特征矩阵，并用残差特征矩阵进行异常检测和诊断" ID="tYaDKoNid5" STYLE="fork"/>
                  </node>
                </node>
                <node TEXT="实验结果" ID="zd4JIZFBwH" STYLE="fork">
                  <node TEXT="数据集：合成数据集、电厂数据集" ID="v3Fc8ZeBME" STYLE="fork"/>
                  <node TEXT="与其他模型比较，MSCRED模型的异常检测性能是最佳的" ID="dNwrKCvrpb" STYLE="fork"/>
                  <node TEXT="根本原因检测" ID="QUeo5UYHkZ" STYLE="fork">
                    <node TEXT="主要对比了MSCRED和LSTM-ED的性能，MSCRED将所有时间序列按其异常得分进行排序，将得分最高的k个序列作为根本原因。" ID="OADzuhNHLY" STYLE="fork"/>
                    <node TEXT="结果显示，MSCRED的表现比LSTMED高出约25.9%。" ID="siOLdIm35i" STYLE="fork"/>
                  </node>
                </node>
              </node>
              <node TEXT="Outlier Detection for Time Series with Recurrent Auto-encoder Ensembles （young young）" ID="EdkhovCKtE" STYLE="fork">
                <node TEXT="论文主题与贡献" ID="GmZOF6zDBo" STYLE="fork">
                  <node TEXT="解决某些情况下自动编码器对异常值过度拟合造成对整体质量的影响" ID="MYu931UHMD" STYLE="fork"/>
                </node>
                <node TEXT="论文中提出的理论模型" ID="i7UhRi0pna" STYLE="fork">
                  <node TEXT=" 集成自编码器" ID="topIWurYUz" STYLE="fork">
                    <node TEXT="目的与优势" ID="65NBjgdXfG" STYLE="fork">
                      <node TEXT="减小由于单个自动编码器产生的偶然误差所带来的误差" ID="uXUvlPzuMB" STYLE="fork"/>
                      <node TEXT="提高基于自编码器的异常检测的准确性" ID="pGdSTE70M8" STYLE="fork"/>
                    </node>
                    <node TEXT="实际使用" ID="X6NBkBaFFw" STYLE="fork">
                      <node TEXT="对每个自动编码器随机删除一些连接得到稀疏自编码器" ID="YQM6Dg7GUX" STYLE="fork"/>
                      <node TEXT=" 降低整体重建误差的方差，提高结果准确性" ID="R7EE57mYx6" STYLE="fork"/>
                    </node>
                    <node TEXT=" 缺点与劣势" ID="ev61EtRk79" STYLE="fork">
                      <node TEXT="只能用于非序列数据" ID="Nr0PfSrRfz" STYLE="fork"/>
                      <node TEXT=" 不能直接用于基于时间序列的异常检测" ID="ldqPvDkRtj" STYLE="fork"/>
                    </node>
                  </node>
                  <node TEXT="S-RNN集成自编码器" ID="bNcDYS3nW7" STYLE="fork">
                    <node TEXT="IF独立框架" ID="YS3sntUaih" STYLE="fork">
                      <node TEXT="模型描述" ID="Z4qObyyyDZ" STYLE="fork">
                        <node TEXT="集成包含多个S-RNN自编码器" ID="Rr1Q9lEAhR" STYLE="fork"/>
                        <node TEXT="每个自编码器由一个编码器和一个解码器组成" ID="33GZMCaAf8" STYLE="fork"/>
                        <node TEXT="每个自动编码器有其不同的稀疏权值向量" ID="VA3M2IkjdK" STYLE="fork"/>
                      </node>
                      <node TEXT="模型结构" ID="bIpL8Qu5XT" STYLE="fork"/>
                    </node>
                  </node>
                  <node TEXT="SF共享框架" ID="OQJZywNTg5" STYLE="fork">
                    <node TEXT="共享目的" ID="WzPnrFvmcd" STYLE="fork">
                      <node TEXT="充分考虑各自编码器加的关联于共性" ID="1YiXSPTEQs" STYLE="fork"/>
                    </node>
                    <node TEXT="模型描述" ID="ITwSKox4Qv" STYLE="fork">
                      <node TEXT="每个自编码器模块分别重构原始时间序列" ID="tGMHKwHXAe" STYLE="fork"/>
                      <node TEXT="使用共享层来完成各模块间的交互" ID="U4xUFn7QHh" STYLE="fork"/>
                    </node>
                    <node TEXT="模型结构" ID="XGDkDWsSOB" STYLE="fork"/>
                    <node TEXT="共享优势" ID="L8lH6Yu2NT" STYLE="fork">
                      <node TEXT="共享层中的参数起到使共享状态稀疏的作用" ID="SpeiSZrvKu" STYLE="fork"/>
                      <node TEXT="避免过度拟合，提高系统鲁棒性" ID="IFCJO1nkhL" STYLE="fork"/>
                    </node>
                  </node>
                </node>
                <node TEXT="论文中的相关实验" ID="psFE66pOhG" STYLE="fork">
                  <node TEXT="使用数据集" ID="N9WZI7byWH" STYLE="fork">
                    <node TEXT="单变量数据集" ID="qSU02MJqzR" STYLE="fork">
                      <node TEXT="NAB" ID="ZsFzwZZndB" STYLE="fork"/>
                    </node>
                    <node TEXT="多变量数据集" ID="ksy8bRgvyj" STYLE="fork">
                      <node TEXT="ECG" ID="rYnTxJXmBl" STYLE="fork"/>
                    </node>
                  </node>
                  <node TEXT="对照实验方法" ID="2iPOXZbmQO" STYLE="fork">
                    <node TEXT="现有解决方案" ID="rkVepgcDYV" STYLE="fork">
                      <node TEXT="LOF、SVM" ID="1Ua3GEXCbh" STYLE="fork"/>
                      <node TEXT="ISF、MP、RN" ID="Muk9bMXNmx" STYLE="fork"/>
                      <node TEXT="CNN、LSTM" ID="gCjTt0ZwHi" STYLE="fork"/>
                    </node>
                  </node>
                  <node TEXT="实验的具体实现" ID="05hpcgOKie" STYLE="fork">
                    <node TEXT="Python 3" ID="KQIJ5GPjM6" STYLE="fork"/>
                    <node TEXT="Tenserflow 1" ID="zMWMZdhL7l" STYLE="fork"/>
                    <node TEXT="Scikit-learn 1" ID="4w9Wyc67ww" STYLE="fork"/>
                  </node>
                  <node TEXT="评价机制" ID="yPqObuPxCu" STYLE="fork">
                    <node TEXT="评价原则" ID="r8QsyVTgDZ" STYLE="fork">
                      <node TEXT="不依赖具体阈值" ID="2XVyc9gKqj" STYLE="fork"/>
                      <node TEXT="反映真阳性、真阴性、假阳性、假阴性全面权衡" ID="GcekM2Ex1z" STYLE="fork"/>
                    </node>
                    <node TEXT="具体机制" ID="pSQux93sGE" STYLE="fork">
                      <node TEXT="PR-AUC" ID="EpcAT49Z9Q" STYLE="fork"/>
                      <node TEXT="ROC-AUC" ID="bcEO1tLsjo" STYLE="fork"/>
                    </node>
                  </node>
                  <node TEXT="实验结果" ID="tHNLV3VFcR" STYLE="fork">
                    <node TEXT="本文提出模型与既有模型的比较" ID="uu9OxXFJWJ" STYLE="fork">
                      <node TEXT="深度学习方法效果更优" ID="LjkNi6ttj2" STYLE="fork"/>
                      <node TEXT="本文方法较其他集成自编码器更适用于序列数据" ID="1HgLwtbjL7" STYLE="fork"/>
                      <node TEXT="集成方法优于大多数单独方法" ID="mpwRxJJlqO" STYLE="fork"/>
                    </node>
                    <node TEXT="自编码器数量的影响" ID="nF9EBDildg" STYLE="fork">
                      <node TEXT="自编码器数量增加，结果更优" ID="9rZMFTrkw6" STYLE="fork"/>
                    </node>
                    <node TEXT="本文两种集成方法的比较" ID="1dON8RQz4N" STYLE="fork">
                      <node TEXT="共享框架性能更好" ID="Q6ceVcZ6iP" STYLE="fork"/>
                      <node TEXT="独立框架内存消耗较少" ID="UEvQINJ41t" STYLE="fork"/>
                    </node>
                  </node>
                </node>
              </node>
            </node>
          </node>
        </node>
        <node TEXT="整理数据集（Kerr）" ID="cBtN01tSnM" STYLE="fork">
          <node TEXT="REFIT 电⽓负载测量数据集 " ID="9y2MDlAaFh" STYLE="fork"/>
          <node TEXT="电⼒负荷图2011-2014数据集" ID="luwjSyypqX" STYLE="fork"/>
          <node TEXT="Numenta异常基准 (NAB) 数据集 " ID="VX9Axg3kyC" STYLE="fork"/>
          <node TEXT="加利福尼亚交通运输数据集 (PeMS数据集) " ID="xgCE8ppxWD" STYLE="fork"/>
          <node TEXT="Yahoo&apos;s Webscope S5 数据集 " ID="BWCQysAYEq" STYLE="fork"/>
          <node TEXT="2018 AIOps&apos;s KPI-Anomaly-Detection 数据集 " ID="Rq4OnyEAaV" STYLE="fork"/>
          <node TEXT="ECG数据集" ID="G5zWefFX3u" STYLE="fork"/>
        </node>
        <node TEXT="搜索模型" ID="8ssjXtumEZ" STYLE="fork">
          <node TEXT="邱博模型（yuan）" ID="Hd8393jnL1" STYLE="fork">
            <node TEXT="数据集" ID="d2CmJfAYTH" STYLE="fork">
              <node TEXT="aiopsdata" ID="VCeXAChkl3" STYLE="fork"/>
              <node TEXT="单变量" ID="DyQtEFOYIb" STYLE="fork"/>
            </node>
            <node TEXT="模型概述（ConvLSTM）" ID="wiQKaTmijh" STYLE="fork">
              <node TEXT="构建" ID="HkFQE2Cbgk" STYLE="fork"/>
              <node TEXT="ConvLSTM就是在LSTM之前加卷积操作，邱博的模型架构为三层卷积池化+LSTM+softmax" ID="2ekArblOPS" STYLE="fork"/>
              <node TEXT="训练时，训练数据以窗口的形式传到模型里进行训练" ID="exZJ7XWv92" STYLE="fork"/>
            </node>
            <node TEXT="运行结果" ID="Jl4xdsV82B" STYLE="fork"/>
          </node>
          <node TEXT="CNN模型（yuan）" ID="Nmrb0NsRRx" STYLE="fork">
            <node TEXT="CNN简述" ID="CqMlciJrCz" STYLE="fork">
              <node TEXT="框架结构" ID="1NLSkjVeXU" STYLE="fork">
                <node TEXT="卷积层【提取特征】" ID="LrUIilri4X" STYLE="fork"/>
                <node TEXT="池化层【降维，减少运算，避免过拟合】" ID="iNFtD7KYwW" STYLE="fork"/>
                <node TEXT="全连接层【分类】" ID="9kyfJ8A8Nz" STYLE="fork"/>
              </node>
              <node TEXT="常见运用" ID="lIbpXvXnn9" STYLE="fork">
                <node TEXT="适合用于具有局部空间相关性的数据，经常用于图像的特征提取和分类等" ID="Y4NCZ4hmmO" STYLE="fork"/>
              </node>
              <node TEXT="CNN用于时间序列异常检测" ID="mOJYxTSF0r" STYLE="fork">
                <node TEXT="不是很合适，CNN进行空间扩展，输入和输出都是静态的，没有划分独立文本的能力。因为时间序列中序列的时间性也体现在位置上，所以可以用CNN。做时间序列异常检测时经常将CNN与RNN结合使用，处理图像时间序列。" ID="jWUWwLKwpc" STYLE="fork"/>
              </node>
            </node>
            <node TEXT="模型介绍" ID="Lsfx8cG21K" STYLE="fork">
              <node TEXT="数据集" ID="IlQFXSkMZL" STYLE="fork"/>
              <node TEXT="构建和训练" ID="vwXitYalWI" STYLE="fork">
                <node TEXT="构建的是一个二分类分类器，经过了两层卷积池化和三层全连接。" ID="TNASWWt6kQ" STYLE="fork"/>
                <node TEXT="模型训练过程中需要进行反向传播和更新模型参数，然后计算每一次训练完成后的准确度并显示训练结果。" ID="auu9E6zVDg" STYLE="fork"/>
              </node>
              <node TEXT="验证" ID="N2axxpYaLO" STYLE="fork">
                <node TEXT="验证数据输入到模型之中，但是仅仅通过模型的计算给出一个判断结果，不改变模型的参数（即不进行反向传播和梯度计算）等。" ID="929DBcOJHd" STYLE="fork"/>
                <node TEXT="根据模型的判断结果和实际值的差别，可以得出模型在验证集上的准确度。" ID="gU2OTSiGkA" STYLE="fork"/>
                <node TEXT="本模型采用了ROC曲线可视化模型的性能。" ID="N0fBWlqOJh" STYLE="fork">
                  <node TEXT="ROC曲线：ROC的全称是Receiver Operating Characteristic Curve，中文名字叫“受试者工作特征曲线”，图样示例如下" ID="POvi1o2wHx" STYLE="fork"/>
                  <node TEXT="该曲线的横坐标为假阳性率（False Positive Rate, FPR），N是真实负样本的个数，FP是N个负样本中被分类器预测为正样本的个数。" ID="cmT6E06DjA" STYLE="fork"/>
                  <node TEXT="纵坐标为真阳性率（True Positive Rate, TPR）， " ID="Ghmsoyj7hH" STYLE="fork"/>
                  <node TEXT="P是真实正样本的个数，TP是P个正样本中被分类器预测为正样本的个数。" ID="o7zvl3foXO" STYLE="fork"/>
                  <node TEXT="通过曲线可以对模型有一个定性的分析，如果要对模型进行量化的分析，此时需要引入一个新的概念，就是AUC（Area under roc Curve）面积，即曲线沿着横轴积分的值。" ID="GOrC5l5uPT" STYLE="fork"/>
                </node>
              </node>
              <node TEXT="运行结果" ID="k5hO1M8Wos" STYLE="fork"/>
            </node>
          </node>
          <node TEXT="RNN模型（yuan）" ID="xWTzCSIrBb" STYLE="fork">
            <node TEXT="结构：RNN中的每个节点都有关联，如下图所示，Xt表示t时刻的输入，Ot是t时刻对应的输出， St是t时刻的存储记忆。对于RNN中的每个单元，输入分为两个部分：" ID="iUB3J1jcU2" STYLE="fork">
              <node TEXT="1）当前时刻的真正的输入Xt；" ID="CR3H1aNQ0f" STYLE="fork"/>
              <node TEXT="2）前一时刻的存储记忆St-1。" ID="Y2KPmR7lNM" STYLE="fork"/>
            </node>
            <node TEXT="常见运用：RNN 常用于序列是相互依赖的（有限或无限）数据流，所以适合时间序列的数据，它的输出可以是一个序列值或者一序列的值。" ID="qlxo2i1eeS" STYLE="fork"/>
            <node TEXT="在时间序列检测中的应用" ID="orHgyO0QqE" STYLE="fork">
              <node TEXT="数据集" ID="1tl1Zqe41R" STYLE="fork">
                <node TEXT="ECG变量" ID="pHT3DNMYkC" STYLE="fork"/>
                <node TEXT="gesture(双变量)" ID="wouffe2UXH" STYLE="fork"/>
                <node TEXT="nyc_taxi(三变量)" ID="CzBpVucyox" STYLE="fork"/>
                <node TEXT="power_demand(单变量)" ID="Kh5tG8AmrT" STYLE="fork"/>
                <node TEXT="respiration(单变量)" ID="F5x53JTjFB" STYLE="fork"/>
                <node TEXT="space_shuttle(单变量)" ID="LUEVsJ6WfO" STYLE="fork"/>
              </node>
              <node TEXT="模型构建及训练" ID="btBEtQ6hBu" STYLE="fork"/>
            </node>
          </node>
          <node TEXT="针对ECG数据的TSI模型（Kerr）" ID="sz9MsnJXGc" STYLE="fork">
            <node TEXT="Github  https://github.com/giorgiodema/ECG-Anomaly-Detection" ID="SBa8Wh7rwD" STYLE="fork"/>
            <node TEXT="Dataset" ID="zQ5RhHSgPd" STYLE="fork">
              <node TEXT="File format " ID="Y2d2jFGuZS" STYLE="fork">
                <node TEXT=".csv" ID="1NrXnF5Dsp" STYLE="fork"/>
              </node>
              <node TEXT="Include" ID="jjWXBB0skI" STYLE="fork">
                <node TEXT="normal.csv" ID="kWeFM4MHrH" STYLE="fork"/>
                <node TEXT="abnormal.csv" ID="qiGYD6oDO7" STYLE="fork"/>
              </node>
              <node TEXT="Data format" ID="8G8K9kvnp5" STYLE="fork">
                <node TEXT="Each time serie has the same lenght (188) and each value isnormalized in the [0,1] range." ID="kmyrECCSbb" STYLE="fork"/>
                <node TEXT="normal data" ID="eXGGVLRZKd" STYLE="fork"/>
                <node TEXT="abnormal data" ID="TABKPuT6By" STYLE="fork"/>
              </node>
            </node>
            <node TEXT="Data process" ID="RaXY5m4Ssz" STYLE="fork">
              <node TEXT="1. balance the data  --  make the scale of normal and abnormal data the same" ID="Vj8QntR0JE" STYLE="fork"/>
              <node TEXT="2. split train set and validation set" ID="ztoP388gz8" STYLE="fork">
                <node TEXT="train set : [0: int(normal.shape[0]*0.8]" ID="uQhmsvLsDN" STYLE="fork"/>
                <node TEXT="validation set : [int(normal.shape[0]*0.8:]" ID="4SsGzuzwvS" STYLE="fork"/>
              </node>
              <node TEXT="3. add label" ID="iqKiFdJUZV" STYLE="fork">
                <node TEXT="[1, 0] label means normal" ID="2xf6JQhtGi" STYLE="fork"/>
                <node TEXT="[0, 1] label means abnormal" ID="d0kvh1kw5X" STYLE="fork"/>
              </node>
              <node TEXT="4. shuffle" ID="1rLOd6WKdT" STYLE="fork"/>
              <node TEXT="5. Time Series to Images" ID="YAPoThzvK7" STYLE="fork">
                <node TEXT="First channel -- Gramian Summation Angular Field Transform " ID="om48RUs80P" STYLE="fork"/>
                <node TEXT="Second channel -- Gramian Difference Angular Field Tronform" ID="4yQaBVVxWb" STYLE="fork"/>
                <node TEXT="Third channel -- Markov Transition Field Transform" ID="MqtXQSBLVA" STYLE="fork"/>
              </node>
            </node>
            <node TEXT="Model" ID="FCbEAGBkcR" STYLE="fork">
              <node TEXT="supervised" ID="fWVHBzS5Va" STYLE="fork">
                <node TEXT="CNN" ID="ZUYcNREMmA" STYLE="fork">
                  <node TEXT="(Conv2D(activation -- relu) -&gt; MaxPool2D)*3 " ID="qMrHTevciT" STYLE="fork"/>
                  <node TEXT="Conv2D(activation -- relu)" ID="1lcfKhULKw" STYLE="fork"/>
                  <node TEXT="Flatten" ID="SzgrbBQ7In" STYLE="fork"/>
                  <node TEXT="Dense(activation -- softmax)" ID="BCAU6ZrCzI" STYLE="fork"/>
                </node>
              </node>
              <node TEXT="unsupervised" ID="clvm9GAhjs" STYLE="fork">
                <node TEXT="GAN" ID="OUsb4OzKAv" STYLE="fork">
                  <node TEXT="encoder" ID="w1gLJIxP9B" STYLE="fork"/>
                  <node TEXT="decoder" ID="7nSCWoyOqp" STYLE="fork"/>
                </node>
              </node>
            </node>
            <node TEXT="Result" ID="SI82kYKYNe" STYLE="fork">
              <node TEXT="supervised" ID="RPK6qwXe56" STYLE="fork">
                <node TEXT="accuracy : 95 %" ID="xPyZAJ5fe7" STYLE="fork"/>
                <node TEXT="precision : 94 %" ID="rNuMiphV3M" STYLE="fork"/>
                <node TEXT="recall    : 95 %" ID="0gDmR5bqP2" STYLE="fork"/>
              </node>
              <node TEXT="unsupervised" ID="BdXlHsnUEj" STYLE="fork">
                <node TEXT="accuracy : 88 %" ID="i2uYg0JDPr" STYLE="fork"/>
                <node TEXT="precision : 85 %" ID="bJiRV7XsLU" STYLE="fork"/>
                <node TEXT="recall: 92 %" ID="C7kJsJv3sD" STYLE="fork"/>
              </node>
            </node>
            <node TEXT="Requirements" ID="tSXR7wAc8n" STYLE="fork">
              <node TEXT="Python 3.7.5" ID="DYMed6TYfz" STYLE="fork"/>
              <node TEXT="tensorflow 2.3.0" ID="sxjRJPljVw" STYLE="fork"/>
              <node TEXT="pyts 0.11.0" ID="ChYjckdhCE" STYLE="fork"/>
            </node>
          </node>
        </node>
      </node>
      <node TEXT="实施阶段" ID="ME3MUyUoLE" STYLE="fork">
        <node TEXT="构建模型（Kerr）" ID="kGR2uhKQw2" STYLE="fork">
          <node TEXT="在服务器上配置环境" ID="zyFBvSjU8f" STYLE="fork"/>
          <node TEXT="训练模型" ID="Tf0UEHukeo" STYLE="fork"/>
          <node TEXT="测试模型" ID="6lJ2t80Eko" STYLE="fork"/>
        </node>
        <node TEXT="设计前端（young young）" ID="quOectnwJX" STYLE="fork">
          <node TEXT="前端设计框架" ID="8nOPinKdPv" STYLE="fork">
            <node TEXT="Vue + Element UI" ID="q38AVXeEJc" STYLE="fork"/>
          </node>
          <node TEXT="设计功能点" ID="TiA1ATHw8L" STYLE="fork">
            <node TEXT="选择算法" ID="hH6E7HZRlB" STYLE="fork"/>
            <node TEXT="选择模型" ID="lffvGg2dDN" STYLE="fork"/>
            <node TEXT="选择数据段" ID="0I8Qqpwgu0" STYLE="fork"/>
            <node TEXT="用图表展示数据集以及异常区域" ID="YwxTxJ0mnN" STYLE="fork"/>
            <node TEXT="展示结论（准确率与检测结论（真假阴阳性））" ID="j4uPqEunGu" STYLE="fork"/>
          </node>
        </node>
        <node TEXT="前后端连接（young young）" ID="ix7YQTjP17" STYLE="fork">
          <node TEXT="使用框架" ID="qVNouAnw1z" STYLE="fork">
            <node TEXT="Django" ID="10ouYB3eBt" STYLE="fork"/>
          </node>
          <node TEXT="数据交互技术" ID="MvfL9aAj0Q" STYLE="fork">
            <node TEXT="JQuery" ID="67IzCkLuK2" STYLE="fork"/>
            <node TEXT="Ajax" ID="p2Aop2pDoN" STYLE="fork"/>
            <node TEXT="JSON" ID="VY1ZybigOW" STYLE="fork"/>
          </node>
        </node>
      </node>
      <node TEXT="测试阶段(Undo)" ID="4VB0CGQfBk" STYLE="fork">
        <node TEXT="功能测试" ID="VhYeeQSPHu" STYLE="fork"/>
        <node TEXT="性能测试" ID="gif46Xb8Rd" STYLE="fork"/>
      </node>
      <node TEXT="发布阶段(Undo)" ID="HFGK0a3CAI" STYLE="fork">
        <node TEXT="整理代码" ID="GwYXFBSSi1" STYLE="fork"/>
        <node TEXT="完善文档" ID="c0mKvkRAQc" STYLE="fork"/>
      </node>
    </node>
    <node TEXT="实用工具" ID="YcwzR9ZDNf" STYLE="bubble" POSITION="left">
      <node TEXT="Microsoft Todo" ID="vL3Stp9LDW" STYLE="fork"/>
      <node TEXT="幕布" ID="xWAb7ODk0x" STYLE="fork"/>
      <node TEXT="Postman" ID="kA21C0HG1c" STYLE="fork"/>
      <node TEXT="Github" ID="oQ5Z4JbaLX" STYLE="fork"/>
    </node>
  </node>
</map>