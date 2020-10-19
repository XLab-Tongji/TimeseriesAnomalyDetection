# Generative Adversarial Networks (GAN) example in PyTorch. Tested with PyTorch 0.4.1, Python 3.6.7 (Nov 2018)

# GAN example in PyTorch. Retested with Pytorch 1.6.1, Python 3.6.8 (Oct 2020)
# 尝试进行20000次epochs后，观察生成图像进行分析
# By 1850250 赵浠明

# GAN的组成部分：
# R：真实数据集
# I：作为熵的一项来源，进入生成器的随机噪声
# G：生成器，试图模仿原始数据
# D：判别器，试图区别G和R

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

matplotlib_is_available = True
try:
    from matplotlib import pyplot as plt
except ImportError:
    print("Will skip plotting; matplotlib is not available.")
    matplotlib_is_available = False

# Data params 输入数据R的属性
data_mean = 4  # 数据的平均值
data_stddev = 1.25  # 数据标准差

# ### Uncomment only one of these to define what data is actually sent to the Discriminator
# (name, preprocess, d_input_func) = ("Raw data", lambda data: data, lambda x: x)
# (name, preprocess, d_input_func) = ("Data and variances", lambda data: decorate_with_diffs(data, 2.0), lambda x: x *2)
# (name, preprocess, d_input_func) = ("Data and diffs", lambda data: decorate_with_diffs(data, 1.0), lambda x: x * 2)
(name, preprocess, d_input_func) = ("Only 4 moments", lambda data: get_moments(data), lambda x: 4)

print("Using data [%s]" % name)


# ##### DATA: Target data and generator input data

# 真实数据R的生成，使用numpy生成高斯分布
def get_distribution_sampler(mu, sigma):
    return lambda n: torch.Tensor(np.random.normal(mu, sigma, (1, n)))  # Gaussian


# 噪声数据I直接使用随机函数生成
def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)  # Uniform-dist data into generator, _NOT_ Gaussian


# ##### MODELS: Generator model and discriminator model

class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Generator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.f(x)
        x = self.map2(x)
        x = self.f(x)
        x = self.map3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(x))


# 提取数据
def extract(v):
    return v.data.storage().tolist()


# 获取数据的平均值和标准差
def stats(d):
    return [np.mean(d), np.std(d)]


#
def get_moments(d):
    # Return the first 4 moments of the data provided
    mean = torch.mean(d)
    diffs = d - mean
    var = torch.mean(torch.pow(diffs, 2.0))
    std = torch.pow(var, 0.5)  # 标准差
    zscores = diffs / std
    skews = torch.mean(torch.pow(zscores, 3.0))  # 不对称度
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0  # 峰度excess kurtosis, should be 0 for Gaussian 峰度
    final = torch.cat((mean.reshape(1, ), std.reshape(1, ), skews.reshape(1, ), kurtoses.reshape(1, )))
    return final


def decorate_with_diffs(data, exponent, remove_raw_data=False):
    mean = torch.mean(data.data, 1, keepdim=True)
    mean_broadcast = torch.mul(torch.ones(data.size()), mean.tolist()[0][0])
    diffs = torch.pow(data - Variable(mean_broadcast), exponent)
    if remove_raw_data:
        return torch.cat([diffs], 1)
    else:
        return torch.cat([data, diffs], 1)


def train():
    # Model parameters
    g_input_size = 1  # Random noise dimension coming into generator, per output vector
    g_hidden_size = 5  # Generator complexity
    g_output_size = 1  # Size of generated output vector 生成的输出向量大小
    d_input_size = 500  # Minibatch size - cardinality of distributions 基数的分布
    d_hidden_size = 10  # Discriminator complexity 分辨器的复杂度
    d_output_size = 1  # Single dimension for 'real' vs. 'fake' classification
    minibatch_size = d_input_size

    d_learning_rate = 1e-3
    g_learning_rate = 1e-3
    sgd_momentum = 0.9

    num_epochs = 20000  # 训练数据过多少遍
    print_interval = 100
    d_steps = 20
    g_steps = 20

    dfe, dre, ge = 0, 0, 0
    d_real_data, d_fake_data, g_fake_data = None, None, None

    discriminator_activation_function = torch.sigmoid  # 激活函数
    generator_activation_function = torch.tanh  # 双曲正切

    d_sampler = get_distribution_sampler(data_mean, data_stddev)
    gi_sampler = get_generator_input_sampler()
    G = Generator(input_size=g_input_size,
                  hidden_size=g_hidden_size,
                  output_size=g_output_size,
                  f=generator_activation_function)
    D = Discriminator(input_size=d_input_func(d_input_size),
                      hidden_size=d_hidden_size,
                      output_size=d_output_size,
                      f=discriminator_activation_function)

    # 二进制交叉熵损失函数，表征真实样本标签和预测概率之间的差值
    criterion = nn.BCELoss()  # Binary cross entropy: http://pytorch.org/docs/nn.html#bceloss

    # 神经网络优化器，实现随机梯度下降算法
    # 获取参数（必须是variable对象），lr是学习率，momentum是“冲量”
    # lr小，收敛到极值的速度较慢
    # momentum在梯度下降时，在公式中作为一个系数
    d_optimizer = optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)
    g_optimizer = optim.SGD(G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)

    for epoch in range(num_epochs):
        # 一遍epoch过20次D
        for d_index in range(d_steps):
            # 1. Train D on real+fake
            D.zero_grad()

            #  1A: Train D on real
            d_real_data = Variable(d_sampler(d_input_size))  # 获取真实数据
            d_real_decision = D(preprocess(d_real_data))  # 获取处理后的数据
            d_real_error = criterion(d_real_decision, Variable(torch.ones([1, 1])))  # one返回1代表true
            d_real_error.backward()  # compute/store gradients, but don't change params 计算存储梯度，但不改变参数

            #  1B: Train D on fake
            d_gen_input = Variable(gi_sampler(minibatch_size, g_input_size))  # 生成噪声数据I
            d_fake_data = G(d_gen_input).detach()  # 分离数据，防止使用假的数据来训练G
            d_fake_decision = D(preprocess(d_fake_data.t()))  # 获取处理后的数据
            d_fake_error = criterion(d_fake_decision, Variable(torch.zeros([1, 1])))  # zeros返回0代表false
            d_fake_error.backward()
            d_optimizer.step()  # 通过backward函数，只优化D的参数

            # 输出d_real_error和d_fake_error，提取出第一个
            dre, dfe = extract(d_real_error)[0], extract(d_fake_error)[0]

        # 一遍epoch过20遍G
        for g_index in range(g_steps):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            # 这时候只通过D的输出来训练G，而不训练D
            G.zero_grad()

            gen_input = Variable(gi_sampler(minibatch_size, g_input_size))  # 生成噪声数据I
            g_fake_data = G(gen_input)  # 生成虚假数据
            dg_fake_decision = D(preprocess(g_fake_data.t()))  # 讲G生成的数据g_fake_data穿过D
            g_error = criterion(dg_fake_decision, Variable(torch.ones([1, 1])))  # 训练G来让它假装自己是真的

            g_error.backward()
            g_optimizer.step()  # Only optimizes G's parameters 只优化G的参数

            # 输出
            ge = extract(g_error)[0]

        if epoch % print_interval == 0:
            print("Epoch %s: D (%s real_err, %s fake_err) G (%s err); Real Dist (%s),  Fake Dist (%s) " %
                  (epoch, dre, dfe, ge, stats(extract(d_real_data)), stats(extract(d_fake_data))))
            # dre D对于真实数据的损失函数
            # dfe D对于虚假数据的损失函数
            # ge  G生成的虚假数据的函数损失
            # stats函数 -> 自己写的，返回数据的平均值和方差

    if matplotlib_is_available:
        print("Plotting the generated distribution...")
        values = extract(g_fake_data)
        print(" Values: %s" % (str(values)))
        plt.hist(values, bins=50)
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.title('Histogram of Generated Distribution')
        plt.grid(True)
        plt.show()


train()
