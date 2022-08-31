from DataLoader import ImageDataset
from Model import Net
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sch
from torch.utils.data import DataLoader
from torchvision import transforms
import warnings
import yaml

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':

    torch.manual_seed(51)

    # Read Parameters.yml file
    with open("../Parameters.yml", 'r') as stream:
        dic = yaml.safe_load(stream)

    dicOptimizerParameters = dic['OptimizerParameters']

    # Paths
    trainPath = dic['Paths']['TrainData']
    modelPath = dic['Paths']['Model']
    saveModel = dic['SaveModel']

    batchSize = dicOptimizerParameters['BatchSize']
    numWorkers = dic['NumberOfWorkers']

    # Image Transformations
    imageTransform1 = transforms.Compose(
        [transforms.RandomEqualize(p=1), transforms.Pad(padding=(3, 30, 30, 30)), transforms.Resize((128, 128)),
         transforms.ToTensor()])
    imageTransform2 = transforms.Compose(
        [transforms.RandomEqualize(p=1), transforms.RandomHorizontalFlip(p=1), transforms.Pad(padding=(3, 30, 30, 30)),
         transforms.Resize((128, 128)), transforms.ToTensor()])
    imageTransform3 = transforms.Compose(
        [transforms.RandomEqualize(p=1), transforms.RandomPerspective(distortion_scale=0.6, p=1),
         transforms.Pad(padding=(3, 30, 30, 30)), transforms.Resize((128, 128)), transforms.ToTensor()])
    imageTransform4 = transforms.Compose([transforms.RandomEqualize(p=1), transforms.RandomHorizontalFlip(p=1),
                                          transforms.RandomPerspective(distortion_scale=0.6, p=1),
                                          transforms.Pad(padding=(3, 30, 30, 30)), transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

    # Training Data
    trainDataset = ImageDataset(csvData=trainPath, train=True, validation=True, imageTransform1=imageTransform1,
                                imageTransform2=imageTransform2, imageTransform3=imageTransform3,
                                imageTransform4=imageTransform4)
    trainLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)

    # Validation Data
    validationDataset = ImageDataset(csvData=trainPath, train=False, validation=True, imageTransform1=imageTransform1)
    validationLoader = DataLoader(validationDataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)

    # Store training loss and validation accuracy after each epoch
    lossArray = []
    accuracyArray = []

    # model used
    model = Net()

    # Optimizer Parameters
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=dicOptimizerParameters['LearningRate'],
                           weight_decay=dicOptimizerParameters['WeightDecay'])
    scheduler = sch.StepLR(optimizer, step_size=dicOptimizerParameters['StepSize'],
                           gamma=dicOptimizerParameters['Gamma'])

    bestAccuracy = -1
    epochs = dic['Epochs']

    epochNo = 1
    epochList = []

    while epochNo <= epochs:
        batchNo = 0
        totalLoss = 0
        for batch_idx, sample in enumerate(trainLoader):
            img = sample['images']
            lab = sample['labels']
            images = img.cuda()
            labels = lab.cuda()
            model = model.cuda()
            criteria = criteria.cuda()
            output = model(images)
            labels = labels.long()
            loss = criteria(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
            batchNo += 1

        averageLoss = totalLoss / batchNo
        lossArray.append(averageLoss)

        model.eval()
        numberOfCorrectSamples = 0
        numberOfSamples = 0

        with torch.no_grad():
            for batch_idx, sample in enumerate(validationLoader):
                images = sample['images']
                labels = sample['labels']
                images = images.cuda()
                labels = labels.cuda()
                mo = model.cuda()
                output = mo(images)
                _, pred = output.max(1)
                numberOfSamples += labels.size(0)
                numberOfCorrectSamples += pred.eq(labels).sum().item()

        accuracy = numberOfCorrectSamples / numberOfSamples
        accuracyArray.append(accuracy)

        if accuracy >= bestAccuracy:
            if saveModel:
                torch.save(model.state_dict(), modelPath)
            bestAccuracy = accuracy

        model.train()
        epochList.append(epochNo)
        scheduler.step()

        epochNo += 1
