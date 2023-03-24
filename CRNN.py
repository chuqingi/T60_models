'''
Developer: Jiaxin Li
E-mail: 1319376761@qq.com
Github: https://github.com/chuqingi/T60_models
Description: Model for estimation reverberation time (T60)
Reference:
[1]Online Blind Reverberation Time Estimation Using CRNNs
[2]Blind Reverberation Time Estimation in Dynamic Acoustic Conditions
'''

import torch
import torch.nn as nn


class CRNN(nn.Module):


    def __init__(self):
        super(CRNN, self).__init__()


        self.fc_in = 490  # Flatten
        self.fc_out = 1  # T60
        self.input_size = 10  # The input size fed into the LSTM
        self.hidden_size = 20  # The hidden layer size of the LSTM
        self.seq_len = 98  # [b, c, t, f]      (c * t * f) / self.input_size = 98

        self.conv1 = nn.Conv2d(1, 5, (1, 10), (1, 2))
        self.bn1 = nn.BatchNorm2d(5)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(5, 5, (1, 10), (1, 2))
        self.bn2 = nn.BatchNorm2d(5)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(5, 5, (1, 11), (1, 2))
        self.bn3 = nn.BatchNorm2d(5)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(5, 5, (1, 11), (1, 2))
        self.bn4 = nn.BatchNorm2d(5)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(5, 5, (3, 8), (2, 2))
        self.bn5 = nn.BatchNorm2d(5)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(5, 5, (4, 7), (2, 1))
        self.bn6 = nn.BatchNorm2d(5)
        self.relu6 = nn.ReLU(inplace=True)

        self.lstm = nn.LSTM(self.input_size, 20, 1, batch_first=True)
        self.pool1 = nn.MaxPool2d((2, 2), (2, 2))
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(self.fc_in, self.fc_out)
        self.relu7 = nn.ReLU(inplace=True)


    def forward(self, x):


        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)

        b, c, t, f = x.shape
        x = x.reshape(b, self.seq_len, self.input_size)
        x, (_) = self.lstm(x)
        # b, self.seq_len, self.hidden_size = x.shape
        x = x.reshape(b, c, t, self.seq_len)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = x.view(-1, self.fc_in)
        x = self.fc1(x)
        x = self.relu1(x)

        return x


def check_parameters(net):
    #  Returns module parameters. Mb
    all_params = sum(param.numel() for param in net.parameters())
    return all_params / 10 ** 6


def test_cnn():
    x = torch.randn(1, 1, 21, 1996)
    net = CRNN()
    s = net(x)
    print('Params: ', str(check_parameters(net)) + ' Mb')
    print(x.shape)
    print(s.shape)
    print(net)


if __name__ == '__main__':
    test_cnn()