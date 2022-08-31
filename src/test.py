from DataLoader import ImageDataset
from Model import Net
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

if __name__ == '__main__':

    torch.manual_seed(51)

    # Read Parameters.yml file
    with open("../Parameters.yml", 'r') as stream:
        dic = yaml.safe_load(stream)

    # Paths
    testPath = dic['Paths']['TestData']
    modelPath = dic['Paths']['Model']
    predictionsPath = dic['Paths']['Predictions']

    imageTransform1 = transforms.Compose(
        [transforms.RandomEqualize(p=1), transforms.Pad(padding=(3, 30, 30, 30)), transforms.Resize((128, 128)),
         transforms.ToTensor()])

    testDataset = ImageDataset(csvData=testPath, train=False, validation=False, imageTransform1=imageTransform1)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False, num_workers=0)

    model = Net()
    model.load_state_dict(torch.load(modelPath))
    model.eval()

    testData = np.asarray(pd.read_csv(testPath))
    testSample = []
    predictions = []
    yogaPoses = ['Still', 'TriyakTadasana', 'Natavarasana', 'Pranamasana', 'Santolanasana', 'Virabhadrasana',
                 'Tuladandasana', 'Trikonasana', 'Natarajasana', 'Katichakrasana', 'Utkatasana', 'Vrikshasana',
                 'Ardhachakrasana', 'Tadasana', 'ParivrittaTrikonasana', 'Naukasana', 'Padahastasana', 'Garudasana',
                 'Gorakshasana']
    i = 0
    with torch.no_grad():
        for batch_idx, sample in enumerate(testLoader):
            img = sample['images']
            images = img.cuda()
            mo = model.cuda()
            output = mo(images)
            _, pred = output.max(1)
            pred = pred.tolist()
            s = str(testData[i])
            s = s[2:-2]
            testSample.append(s)
            i += 1
            for ite in pred:
                predictions.append(yogaPoses[int(ite)])

    df = pd.DataFrame(list(zip(testSample, predictions)), columns=['name', 'category'])
    df.to_csv(predictionsPath, index=False)
