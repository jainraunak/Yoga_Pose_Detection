import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

class ImageDataset(Dataset):
    def __init__(self, csvData, train=True, validation=True, imageTransform1=None, imageTransform2=None,
                 imageTransform3=None, imageTransform4=None):
        self.csvData = csvData
        self.imageTransform1 = imageTransform1
        self.imageTransform2 = imageTransform2
        self.imageTransform3 = imageTransform3
        self.imageTransform4 = imageTransform4
        self.isTrain = train
        self.isValidation = validation
        self.yogaPoses = {'Still': 0, 'TriyakTadasana': 1, 'Natavarasana': 2, 'Pranamasana': 3, 'Santolanasana': 4,
                          'Virabhadrasana': 5, 'Tuladandasana': 6, 'Trikonasana': 7, 'Natarajasana': 8,
                          'Katichakrasana': 9,'Utkatasana': 10, 'Vrikshasana': 11, 'Ardhachakrasana': 12,
                          'Tadasana': 13,'ParivrittaTrikonasana': 14, 'Naukasana': 15, 'Padahastasana': 16,
                          'Garudasana': 17,'Gorakshasana': 18}
        data = pd.read_csv(csvData)
        r = int(data.shape[0]/10)

        if self.isTrain:
            images = data.iloc[:, 0].to_numpy()
            labels = data.iloc[:, -1].to_numpy()
            label = np.zeros(4*labels.shape[0], dtype=int)
            image = np.zeros(4*images.shape[0], dtype=object)
            i = 0
            k = 0
            while i < labels.shape[0]:
                label[k] = self.yogaPoses[labels[i]]
                label[k+1] = self.yogaPoses[labels[i]]
                label[k+2] = self.yogaPoses[labels[i]]
                label[k+3] = self.yogaPoses[labels[i]]
                image[k] = images[i]
                image[k+1] = images[i]
                image[k+2] = images[i]
                image[k+3] = images[i]
                k += 4
                i += 1
        elif self.isValidation:
            images = data.iloc[9*r:, 0].to_numpy()
            labels = data.iloc[9*r:, -1].to_numpy()
            label = np.zeros(labels.shape[0], dtype=int)
            image = np.zeros(images.shape[0], dtype=object)
            i = 0
            k = 0
            while i < labels.shape[0]:
                label[k] = self.yogaPoses[labels[i]]
                image[k] = images[i]
                k += 1
                i += 1
        else:
            image = data.iloc[:, 0].to_numpy()
            label = None

        self.images = image
        self.labels = label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Crop the images
        left = 100
        top = 10
        right = 400
        bottom = 298
        img = Image.open(image)
        img = img.crop((left, top, right, bottom))

        if self.isTrain:
            label = self.labels[idx]
        elif self.isValidation:
            label = self.labels[idx]
        else:
            label = -1

        if idx % 4 == 0 or self.isTrain == False:
            image = self.imageTransform1(img)
        elif idx % 4 == 1:
            image = self.imageTransform2(img)
        elif idx % 4 == 2:
            image = self.imageTransform3(img)
        elif idx % 4 == 3:
            image = self.imageTransform4(img)

        sample = {"images": image, "labels": label}

        return sample