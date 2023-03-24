'''
Developer: Jiaxin Li
E-mail: 1319376761@qq.com
Github: https://github.com/chuqingi/T60_models
Description: Model for estimation reverberation time (T60)
Reference: Impulse response data augmentation and deep neural networks for blind room acoustic parameter estimation
'''

import torch
import torch.nn as nn


class CNN(nn.Module):


    def __init__(self):
        super(CNN, self).__init__()


        self.fc_in = 32*7*6  # Flatten
        self.fc_out = 1  # T60

        self.conv1 = nn.Conv2d(1, 8, (1, 2), (1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d((1, 2), (1, 2))
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 8, (1, 2), (1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d((1, 2), (1, 2))
        self.bn2 = nn.BatchNorm2d(8)

        self.conv3 = nn.Conv2d(8, 16, (1, 2), (1, 1))
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d((1, 2), (1, 2))
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, (1, 2), (1, 1))
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d((1, 2), (1, 2))
        self.bn4 = nn.BatchNorm2d(16)

        self.conv5 = nn.Conv2d(16, 32, (2, 2), (1, 1))
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d((2, 2), (2, 2))
        self.bn5 = nn.BatchNorm2d(32)

        self.conv6 = nn.Conv2d(32, 32, (2, 2), (1, 1))
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool2d((2, 2), (2, 2))
        self.bn6 = nn.BatchNorm2d(32)

        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(self.fc_in, self.fc_out)
        self.relu7 = nn.ReLU(inplace=True)


    def forward(self, x):


        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = self.bn5(x)

        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)
        x = self.bn6(x)

        x = self.dropout1(x)
        x = x.view(-1, self.fc_in)
        x = self.fc1(x)
        x = self.relu7(x)

        return x


def check_parameters(net):
    #  Returns module parameters. Mb
    all_params = sum(param.numel() for param in net.parameters())
    return all_params / 10 ** 6


def test_cnn():
    x = torch.randn(1, 1, 32, 499)
    net = CNN()
    s = net(x)
    print('Params: ', str(check_parameters(net)) + ' Mb')
    print(x.shape)
    print(s.shape)
    print(net)


if __name__ == '__main__':
    test_cnn()