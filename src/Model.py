import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.bn12 = nn.BatchNorm2d(32)
        self.bn13 = nn.BatchNorm2d(32)
        self.bn14 = nn.BatchNorm2d(32)
        self.bn15 = nn.BatchNorm2d(32)
        self.conv12 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same')
        self.conv13 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same')
        self.conv14 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same')
        self.conv15 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.drp1 = nn.Dropout(p=0.5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.bn22 = nn.BatchNorm2d(64)
        self.bn23 = nn.BatchNorm2d(64)
        self.bn24 = nn.BatchNorm2d(64)
        self.bn25 = nn.BatchNorm2d(64)
        self.conv22 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.conv23 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.conv24 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.conv25 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.drp2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.bn32 = nn.BatchNorm2d(128)
        self.bn33 = nn.BatchNorm2d(128)
        self.bn34 = nn.BatchNorm2d(128)
        self.bn35 = nn.BatchNorm2d(128)
        self.conv32 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same')
        self.conv33 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same')
        self.conv34 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same')
        self.conv35 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.drp3 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(8192 * 4, 2048)
        self.bnfc1 = nn.BatchNorm1d(2048)
        self.drpf4 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(2048, 256)
        self.bnfc2 = nn.BatchNorm1d(256)
        self.drpf5 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 19)

    def forward(self, X):
        X = self.bn(X)
        X = F.elu(self.conv1(X))
        X = self.bn1(X)
        X = F.elu(self.conv12(X))
        X = self.bn12(X)
        X = F.elu(self.conv13(X))
        X = self.bn13(X)
        X = F.elu(self.conv14(X))
        X = self.bn14(X)
        X = F.elu(self.conv15(X))
        X = self.bn15(X)
        X = self.pool1(X)
        X = self.drp1(X)
        X = F.elu(self.conv2(X))
        X = self.bn2(X)
        X = F.elu(self.conv22(X))
        X = self.bn22(X)
        X = F.elu(self.conv23(X))
        X = self.bn23(X)
        X = F.elu(self.conv24(X))
        X = self.bn24(X)
        X = F.elu(self.conv25(X))
        X = self.bn25(X)
        X = self.pool2(X)
        X = self.drp2(X)
        X = F.elu(self.conv3(X))
        X = self.bn3(X)
        X = F.elu(self.conv32(X))
        X = self.bn32(X)
        X = F.elu(self.conv33(X))
        X = self.bn33(X)
        X = F.elu(self.conv34(X))
        X = self.bn34(X)
        X = F.elu(self.conv35(X))
        X = self.bn35(X)
        X = self.pool3(X)
        X = self.drp3(X)
        X = X.view(-1, 8192 * 4)
        X = F.elu(self.fc1(X))
        X = self.bnfc1(X)
        X = self.drpf4(X)
        X = F.elu(self.fc2(X))
        X = self.bnfc2(X)
        X = self.drpf5(X)
        X = self.fc3(X)
        return X