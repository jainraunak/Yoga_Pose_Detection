import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, data_csv, train=True, vali=True, img_transform=None, img_transform1=None, img_transform2=None,
                 img_transform3=None):
        self.data_csv = data_csv
        self.img_transforms = img_transform
        self.img_transforms1 = img_transform1
        self.img_transforms2 = img_transform2
        self.img_transforms3 = img_transform3
        self.is_train = train
        self.is_vali = vali
        data = pd.read_csv(data_csv)
        r = data.shape[0]
        r = int(r / 10)
        if self.is_train:
            images = data.iloc[:, 0].to_numpy()
            labels = data.iloc[:, -1].to_numpy()
            lab = {'Still': 0, 'TriyakTadasana': 1, 'Natavarasana': 2, 'Pranamasana': 3, 'Santolanasana': 4,
                   'Virabhadrasana': 5, 'Tuladandasana': 6, 'Trikonasana': 7, 'Natarajasana': 8, 'Katichakrasana': 9,
                   'Utkatasana': 10, 'Vrikshasana': 11, 'Ardhachakrasana': 12, 'Tadasana': 13,
                   'ParivrittaTrikonasana': 14, 'Naukasana': 15, 'Padahastasana': 16, 'Garudasana': 17,
                   'Gorakshasana': 18}
            label = np.zeros(4 * labels.shape[0], dtype=int)
            image = np.zeros(4 * images.shape[0], dtype=object)
            i = 0
            k = 0
            while (i < labels.shape[0]):
                label[k] = lab[labels[i]]
                label[k + 1] = lab[labels[i]]
                label[k + 2] = lab[labels[i]]
                label[k + 3] = lab[labels[i]]
                image[k] = images[i]
                image[k + 1] = images[i]
                image[k + 2] = images[i]
                image[k + 3] = images[i]
                k = k + 4
                i = i + 1
            #print("Done")
        elif (self.is_vali == True):
            images = data.iloc[9 * r:, 0].to_numpy()
            labels = data.iloc[9 * r:, -1].to_numpy()
            lab = {'Still': 0, 'TriyakTadasana': 1, 'Natavarasana': 2, 'Pranamasana': 3, 'Santolanasana': 4,
                   'Virabhadrasana': 5, 'Tuladandasana': 6, 'Trikonasana': 7, 'Natarajasana': 8, 'Katichakrasana': 9,
                   'Utkatasana': 10, 'Vrikshasana': 11, 'Ardhachakrasana': 12, 'Tadasana': 13,
                   'ParivrittaTrikonasana': 14, 'Naukasana': 15, 'Padahastasana': 16, 'Garudasana': 17,
                   'Gorakshasana': 18}
            label = np.zeros(labels.shape[0], dtype=int)
            image = np.zeros(images.shape[0], dtype=object)
            i = 0
            k = 0
            while (i < labels.shape[0]):
                label[k] = lab[labels[i]]
                image[k] = images[i]
                k = k + 1
                i = i + 1
            #print("Done")
        else:
            image = data.iloc[:, 0].to_numpy()
            label = None
        self.images = image
        self.labels = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        left = 100
        top = 10
        right = 400
        bottom = 298
        img = Image.open(image)
        img = img.crop((left, top, right, bottom))
        if self.is_train:
            label = self.labels[idx]
        elif (self.is_vali == True):
            label = self.labels[idx]
        else:
            label = -1
        if (idx % 4 == 0 or self.is_vali == False):
            image = self.img_transforms(img)
        elif (idx % 4 == 1):
            image = self.img_transforms1(img)
        elif (idx % 4 == 2):
            image = self.img_transforms2(img)
        elif (idx % 4 == 3):
            image = self.img_transforms3(img)
        sample = {"images": image, "labels": label}
        return sample


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


if __name__ == '__main__':
    
    path_to_weight_folder = sys.argv[1]
    path_to_testfile = sys.argv[2]
    path_to_testoutput = sys.argv[3]
    
    torch.manual_seed(51)
    BATCH_SIZE = 1
    NUM_WORKERS = 0

    img_transforms = transforms.Compose(
        [transforms.RandomEqualize(p=1), transforms.Pad(padding=(3, 30, 30, 30)), transforms.Resize((128, 128)),
         transforms.ToTensor()])
    img_transforms1 = transforms.Compose(
        [transforms.RandomEqualize(p=1), transforms.RandomHorizontalFlip(p=1), transforms.Pad(padding=(3, 30, 30, 30)),
         transforms.Resize((128, 128)), transforms.ToTensor()])
    img_transforms2 = transforms.Compose(
        [transforms.RandomEqualize(p=1), transforms.RandomPerspective(distortion_scale=0.6, p=1),
         transforms.Pad(padding=(3, 30, 30, 30)), transforms.Resize((128, 128)), transforms.ToTensor()])
    img_transforms3 = transforms.Compose([transforms.RandomEqualize(p=1), transforms.RandomHorizontalFlip(p=1),
                                          transforms.RandomPerspective(distortion_scale=0.6, p=1),
                                          transforms.Pad(padding=(3, 30, 30, 30)), transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

    test_data = path_to_testfile
    test_dataset = ImageDataset(data_csv=test_data, train=False, vali=False, img_transform=img_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    mpth = os.path.join(path_to_weight_folder,'model_SR.pth')
    model = Net()
    model.load_state_dict(torch.load(mpth))
    predic = path_to_testoutput
    model.eval()
    dftest = pd.read_csv(test_data)
    nparr = dftest.values
    lis = []
    ans = []
    arr1 = ['Still', 'TriyakTadasana', 'Natavarasana', 'Pranamasana', 'Santolanasana', 'Virabhadrasana',
            'Tuladandasana', 'Trikonasana', 'Natarajasana', 'Katichakrasana', 'Utkatasana', 'Vrikshasana',
            'Ardhachakrasana', 'Tadasana', 'ParivrittaTrikonasana', 'Naukasana', 'Padahastasana', 'Garudasana',
            'Gorakshasana']
    i = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            img = sample['images']
            images = img.cuda()
            mo = model.cuda()
            output = mo(images)
            _, pred = output.max(1)
            pred = pred.tolist()
            s = str(nparr[i])
            s = s[2:-2]
            lis.append(s)
            i = i + 1
            for ite in pred:
                ans.append(arr1[int(ite)])
    dfans = pd.DataFrame(list(zip(lis, ans)), columns=['name', 'category'])
    dfans.to_csv(predic, index=False)
