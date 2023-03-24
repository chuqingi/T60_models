'''
Developer: Jiaxin Li
E-mail: 1319376761@qq.com
Github: https://github.com/chuqingi/T60_models
Description: Model for estimation reverberation time (T60)
Reference:
[1]Blind Estimation of Room Acoustic Parameters and Speech Transmission Index using MTF-based CNNs
[2]Blind estimation of speech transmission index and room acoustic parameters based on the extended model of room impulse response
'''

import torch
import torch.nn as nn


class CNN(nn.Module):


    def __init__(self):
        super(CNN, self).__init__()


        self.fc_in = 4*86  # Flatten
        self.fc_out = 1  # T60

        self.conv1 = nn.Conv1d(1, 32, 10, 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(2, 1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 16, 5, 1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(2, 1)
        self.dropout1 = nn.Dropout(0.4)

        self.conv3 = nn.Conv1d(16, 8, 5, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.conv4 = nn.Conv1d(8, 4, 5, 1)
        self.fc1 = nn.Linear(self.fc_in, self.fc_out)
        self.relu4 = nn.ReLU(inplace=True)


    def forward(self, x):


        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = x.view(-1, self.fc_in)
        x = self.fc1(x)
        x = self.relu4(x)

        return x


def check_parameters(net):
    #  Returns module parameters. Mb
    all_params = sum(param.numel() for param in net.parameters())
    return all_params / 10 ** 6


def test_cnn():
    x = torch.randn(1, 1, 200)
    net = CNN()
    s = net(x)
    print('Params: ', str(check_parameters(net)) + ' Mb')
    print(x.shape)
    print(s.shape)
    print(net)


if __name__ == '__main__':
    test_cnn()