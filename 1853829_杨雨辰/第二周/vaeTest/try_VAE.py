
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image


# 定义损失函数，计算其中的损失
def loss_function(recon_x, x, mu, logvar):
    """
    :param recon_x: generated image
    :param x: original image
    :param mu: latent mean of z （隐藏层相关）
    :param logvar: latent log variance of z
    """

    BCE_loss = nn.BCELoss(reduction='sum')   # 使用BCE损失函数
    reconstruction_loss = BCE_loss(recon_x, x)
    KL_divergence = -0.5 * torch.sum(1+logvar-torch.exp(logvar)-mu**2)
    #KLD_ele = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    #KLD = torch.sum(KLD_ele).mul_(-0.5)
    print(reconstruction_loss, KL_divergence)

    # 损失包括两个部分：重构的损失和KL散度
    return reconstruction_loss + KL_divergence


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2_mean = nn.Linear(400, 20)
        self.fc2_logvar = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        # relu: 线性整流函数，作为神经元的激活函数
        h1 = F.relu(self.fc1(x))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparametrization(self, mu, logvar):
        # sigma = 0.5*exp(log(sigma^2))= 0.5*exp(log(var))
        std = 0.5 * torch.exp(logvar)
        # N(mu, std^2) = N(0, 1) * std + mu
        z = torch.randn(std.size()) * std + mu
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrization(mu, logvar)
        return self.decode(z), mu, logvar


# transforms.Compose()类：串联多个图片变换的操作
transform = transforms.Compose([
    transforms.ToTensor(),                # 把图片灰度范围从0-255变换到[0, 1]之间
    transforms.Normalize([0.5], [0.5]),   # 将灰度归一化为[-1, 1]
])

# 导入数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

vae = VAE()
optimizer = torch.optim.Adam(vae.parameters(), lr=0.0003)  # 定义优化器，使用Adam优化算法


# Training
def train(epoch):
    vae.train()
    all_loss = 0.

    # 遍历所有的训练数据集
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        # 将输入内容推平为一维向量
        real_imgs = torch.flatten(inputs, start_dim=1)

        # 训练判断器，利用生成的图片去计算损失函数的值
        gen_imgs, mu, logvar = vae(real_imgs)
        loss = loss_function(gen_imgs, real_imgs, mu, logvar)

        optimizer.zero_grad()    # 梯度置零
        loss.backward()          # 反向传播
        optimizer.step()         # 单次优化

        all_loss += loss.item()  # 统计损失
        print('Epoch {}, loss: {:.6f}'.format(epoch, all_loss/(batch_idx+1)))

    # Save generated images for every epoch
    fake_images = gen_imgs.view(-1, 1, 28, 28)
    save_image(fake_images, 'MNIST_FAKE/fake_images-{}.png'.format(epoch + 1))


# 执行若干轮训练（这里定义为30轮，训练时间大约15分钟）
for epoch in range(30):
    train(epoch)

# 存储训练模型
torch.save(vae.state_dict(), './vae.pth')
