# ECG Anomaly Detection
In this project we present two ECG anomaly detection systems using two distinct approaches:
- <b>supervised learning</b>
- <b>unsupervised learning</b>
# Dataset
The dataset consists in two csv files:
- <b>normal.csv</b> containing normal ecgs
- <b>abnormal.csv</b> containing abnormal ecgs

Each time serie has the same lenght (188) and each value is
normalized in the [0,1] range.
# Data representation
In this project we'll use tools to graphically represent a time serie in order to use efficient computer graphics techniques for classification. We encode each time serie <i>s = [x1,...,xn]</i> into a 3-channels image where:
- the first channel is the <b>Gramian Summation Angular Field Transform</b> of the image
- the second channel is the <b>Gramian Difference Angular Field Tronform</b> of the image
- the third channel is the <b>Markov Transition Field Transform</b> of the image

These techniques are described in detail here
[[1]](#1)

# Supervised Learning
## Architecture
In this first approach we'll use a classic convolutional neural network (CNN) with Convolutional layers interleaved by Max Pooling layers. We'll train this model on the labelled images where:
- [1.,0.] label means <i>normal</i>
- [0.,1.] label means <i>abnormal</i>

We used <i>binary cross entropy</i> loss and <i>Adam</i> optimizer.
## Results
After just 10 epochs of training these are the results of the evaluation of the model on the validation set:
- accuracy : 95 %
- precision : 94 %
- recall    : 95 %

# Unsupervised Learning
In this second approach we'll not use the labels. We'll train the model only on the normal samples. Then at evaluation time we evaluate the model also on abnormal samples and we measure how much the model can distinguish normal samples from abnormal samples.
## Architecture
We'll use a model called <b>GANomaly</b>, this model is described here
[[2]](#2)

## Results

These are the values of the evaluation metrics on the validation set:
- accuracy : 88 %
- precision : 85 %
- recall: 92 %

# References
<a id="1">[1]</a>
Zhiguang Wang, Tim Oates.
<b>Imaging Time-Series to Improve Classification and Imputation</b>, <i>2015</i>

<a id="2">[2]</a>
Federico Di Mattia, Paolo Galeone, Michele De Simoni, Emanuele Ghelfi.
<b>A Survey on GANs for Anomaly Detection</b>,<i>2019</i>

# Requirements
- Python 3.7.5
- tensorflow 2.3.0
- pyts 0.11.0
