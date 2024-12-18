#DataManagerPytorch = All special Pytorch functions here. Needs torch, torchvision, math, matplotlib, random, os and PIL
#Current Version Number = 1.1 (July 15, 2022), Please do not remove this comment
#Current supported datasets = CIFAR-10, CIFAR-100, Tiny ImageNet (requires image files and path)
import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math 
import random 
#import matplotlib.pyplot as plt
#import os 
#import PIL
from random import shuffle

#Convert an image dataloader (I) to a repeat encoding dataloader (E)
def DataLoaderIToDataLoaderRE(dataLoaderI, length):
    #First convert the image dataloader to tensor form
    xTensor, yTensor = DataLoaderToTensor(dataLoaderI)
    #Create memory for the new tensor with repeat encoding 
    xTensorRepeat = torch.zeros(xTensor.shape + (length,))
    #Go through and fill in the new array, probably a faster way to do this with Pytorch tensors
    for i in range(0, xTensor.shape[0]):
        for j in range(0, length):
            xTensorRepeat[i, :, :, :, j] = xTensor[i]
    #New tensor is filled in, convert back to dataloader
    dataLoaderRE = TensorToDataLoader(xTensorRepeat, yTensor, transforms=None, batchSize =dataLoaderI.batch_size, randomizer = None)
    return dataLoaderRE

#Convert an image dataloader (I) to a repeat encoding dataloader (E)
def DataLoaderREToDataLoaderI(dataLoaderRE):
    #First convert the repeated dataloader to tensor form
    xTensor, yTensor = DataLoaderToTensor(dataLoaderRE)
    #Create memory for the new tensor with repeat encoding 
    xTensorImages = torch.zeros(xTensor.shape[0], xTensor.shape[1], xTensor.shape[2], xTensor.shape[3])
    #Go through and fill in the new array, probably a faster way to do this with Pytorch tensors
    for i in range(0, xTensor.shape[0]):
        xTensorImages[i] = xTensor[i, :, :, :, 0] #Just take the first image from the repeated tensor because they should be the same
    #New tensor is filled in, convert back to dataloader
    dataLoaderI = TensorToDataLoader(xTensorImages, yTensor, transforms=None, batchSize =dataLoaderRE.batch_size, randomizer = None)
    return dataLoaderI

def CheckCudaMem():
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print("Unfree Memory=", a)

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)

#Validate using a dataloader 
def validateD(valLoader, model, device=None):
    #switch to evaluate mode
    model.eval()
    acc = 0 
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume cuda
                inputVar = input.cpu() #.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

#Method to validate data using Pytorch tensor inputs and a Pytorch model 
def validateT(xData, yData, model, batchSize=None):
    acc = 0 #validation accuracy 
    numSamples = xData.shape[0]
    model.eval() #change to eval mode
    if batchSize == None: #No batch size so we can feed everything into the GPU
         output = model(xData)
         for i in range(0, numSamples):
             if output[i].argmax(axis=0) == yData[i]:
                 acc = acc+ 1
    else: #There are too many samples so we must process in batch
        numBatches = int(math.ceil(xData.shape[0] / batchSize)) #get the number of batches and type cast to int
        for i in range(0, numBatches): #Go through each batch 
            print(i)
            modelOutputIndex = 0 #reset output index
            startIndex = i*batchSize
            #change the end index depending on whether we are on the last batch or not:
            if i == numBatches-1: #last batch so go to the end
                endIndex = numSamples
            else: #Not the last batch so index normally
                endIndex = (i+1)*batchSize
            output = model(xData[startIndex:endIndex])
            for j in range(startIndex, endIndex): #check how many samples in the batch match the target
                if output[modelOutputIndex].argmax(axis=0) == yData[j]:
                    acc = acc+ 1
                modelOutputIndex = modelOutputIndex + 1 #update the output index regardless
    #Do final averaging and return 
    acc = acc / numSamples
    return acc

#Input a dataloader and model
#Instead of returning a model, output is array with 1.0 dentoting the sample was correctly identified
def validateDA(valLoader, model, device=None):
    numSamples = len(valLoader.dataset)
    accuracyArray = torch.zeros(numSamples) #variable for keep tracking of the correctly identified samples 
    #switch to evaluate mode
    model.eval()
    indexer = 0
    accuracy = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume CUDA by default
                inputVar = input.cpu() #.cuda()
            else:
                inputVar = input.to(device) #use the prefered device if one is specified
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    accuracyArray[indexer] = 1.0 #Mark with a 1.0 if sample is correctly identified
                    accuracy = accuracy + 1
                indexer = indexer + 1 #update the indexer regardless 
    accuracy = accuracy/numSamples
    print("Accuracy:", accuracy)
    return accuracyArray

#Replicate TF's predict method behavior 
def predictD(dataLoader, numClasses, model, device=None):
    numSamples = len(dataLoader.dataset)
    yPred = torch.zeros(numSamples, numClasses)
    #switch to evaluate mode
    model.eval()
    indexer = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None:
                inputVar = input.cpu() #.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            for j in range(0, sampleSize):
                yPred[indexer] = output[j]
                indexer = indexer + 1 #update the indexer regardless 
    return yPred

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xData = torch.zeros((numSamples,) + sampleShape) #Make it generic shape for non-image datasets
    yData = torch.zeros(numSamples)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData 

#Get the output shape from the dataloader
def GetOutputShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape

#Returns the train and val loaders  
def LoadFashionMNISTAsPseudoRGB(batchSize):
    #First transformation, just convert to tensor so we can add in the color channels 
    transformA= transforms.Compose([
        transforms.ToTensor(),
    ])
    #Make the train loader 
    trainLoader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='./data', train=True, download=True, transform=transformA), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    numSamplesTrain = len(trainLoader.dataset) 
    sampleIndex = 0
    #This part hard coded for Fashion-MNIST
    xTrain = torch.zeros(numSamplesTrain, 3, 28, 28)
    yTrain = torch.zeros((numSamplesTrain), dtype=torch.long)
    #Go through and process the data in batches 
    for i,(input, target) in enumerate(trainLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xTrain[sampleIndex,0] = input[batchIndex]
            xTrain[sampleIndex,1] = input[batchIndex]
            xTrain[sampleIndex,2] = input[batchIndex]
            yTrain[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    #Make the validation loader 
    valLoader = torch.utils.data.DataLoader(datasets.FashionMNIST(root='./data', train=False, download=True, transform=transformA), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    numSamplesTest = len(valLoader.dataset) 
    sampleIndex = 0 #reset the sample index to use with the validation loader 
    #This part hard coded for Fashion-MNIST
    xTest = torch.zeros(numSamplesTest, 3, 28, 28)
    yTest = torch.zeros((numSamplesTest),dtype=torch.long)
    #Go through and process the data in batches 
    for i,(input, target) in enumerate(valLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xTest[sampleIndex,0] = input[batchIndex]
            xTest[sampleIndex,1] = input[batchIndex]
            xTest[sampleIndex,2] = input[batchIndex]
            yTest[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    transform_train = torch.nn.Sequential(
        transforms.RandomResizedCrop((224, 224), scale=(0.5, 1.0)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    transform_test = torch.nn.Sequential(
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    trainLoaderFinal = TensorToDataLoader(xTrain, yTrain, transform_train, batchSize, True)
    testLoaderFinal = TensorToDataLoader(xTest, yTest, transform_test, batchSize)
    return trainLoaderFinal, testLoaderFinal

#Show 20 images, 10 in first and row and 10 in second row 
def ShowImages(xFirst, xSecond):
    n = 10  # how many digits we will display
    plt.figure(figsize=(5, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(xFirst[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(xSecond[i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

#This method randomly creates fake labels for the attack 
#The fake target is guaranteed to not be the same as the original class label 
def GenerateTargetsLabelRandomly(yData, numClasses):
    fTargetLabels=torch.zeros(len(yData))
    for i in range(0, len(yData)):
        targetLabel=random.randint(0,numClasses-1)
        while targetLabel==yData[i]:#Target and true label should not be the same 
            targetLabel=random.randint(0,numClasses-1) #Keep flipping until a different label is achieved 
        fTargetLabels[i]=targetLabel
    return fTargetLabels

#Return the first n correctly classified examples from a model 
#Note examples may not be class balanced 
def GetFirstCorrectlyIdentifiedExamples(device, dataLoader, model, numSamples):
    #First check how many samples in the dataset
    numSamplesTotal = len(dataLoader.dataset) 
    sampleShape = GetOutputShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xClean = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xClean = torch.zeros((numSamples,) + sampleShape)
    yClean = torch.zeros(numSamples)
    #switch to evaluate mode
    model.eval()
    acc = 0 
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(dataLoader):
            batchSize = input.shape[0] #Get the number of samples used in each batch
            inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, batchSize):
                #Add the sample if it is correctly identified and we are not at the limit
                if output[j].argmax(axis=0) == target[j] and sampleIndex<numSamples: 
                    xClean[sampleIndex] = input[j]
                    yClean[sampleIndex] = target[j]
                    sampleIndex = sampleIndex+1
    #Done collecting samples, time to covert to dataloader 
    cleanLoader = TensorToDataLoader(xClean, yClean, transforms=None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanLoader

#This data is in the range 0 to 1
def GetCIFAR10Validation224(batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR10Validation160(batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((160, 128)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR100Validation(imgSize=224, batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR100(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR100Training(imgSize=224, batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    trainLoader = torch.utils.data.DataLoader(datasets.CIFAR100(root='./data', train=True, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return trainLoader


def GetCIFAR100Validation160(batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((160, 128)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR100(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR10Validation(imgSize = 32, batchSize=128):
    transformTest = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    valLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, download=True, transform=transformTest), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return valLoader

#This data is in the range 0 to 1
def GetCIFAR10Training(imgSize = 32, batchSize=128):
    toTensorTransform = transforms.Compose([
        transforms.Resize((imgSize, imgSize)),
        transforms.ToTensor()
    ])
    trainLoader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True, download=True, transform=toTensorTransform), batch_size=batchSize, shuffle=False, num_workers=1, pin_memory=True)
    return trainLoader

def GetCorrectlyIdentifiedSamplesBalanced(model, totalSamplesRequired, dataLoader, numClasses):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    #correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    correctlyClassifiedSamples = torch.zeros(((numClasses,) + (numSamplesPerClass,) + sampleShape))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = predictD(dataLoader, numClasses, model)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    #xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    xCorrect = torch.zeros(((totalSamplesRequired,) + sampleShape))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

def GetCorrectlyIdentifiedSamplesBalancedDefense(defense, totalSamplesRequired, dataLoader, numClasses, device):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    #Basic error checking 
    if totalSamplesRequired % numClasses != 0:
        raise ValueError("The total number of samples in not evenly divisable by the number of classes.")
    #Get the number of samples needed for each class
    numSamplesPerClass = int(totalSamplesRequired/numClasses) 
    #correctlyClassifiedSamples = torch.zeros((numClasses, numSamplesPerClass, sampleShape[0], sampleShape[1], sampleShape[2]))
    correctlyClassifiedSamples = torch.zeros(((numClasses,) + (numSamplesPerClass,) + sampleShape))
    sanityCounter = torch.zeros((numClasses))
    #yPred = model.predict(xData)
    yPred = defense.predictD(dataLoader, numClasses, device)
    for i in range(0, xData.shape[0]): #Go through every sample 
        predictedClass = yPred[i].argmax(axis=0)
        trueClass = yData[i]#.argmax(axis=0) 
        currentSavedCount = int(sanityCounter[int(trueClass)]) #Check how may samples we previously saved from this class
        #If the network predicts the sample correctly and we haven't saved enough samples from this class yet then save it
        if predictedClass == trueClass and currentSavedCount<numSamplesPerClass:
            correctlyClassifiedSamples[int(trueClass), currentSavedCount] = xData[i] #Save the sample 
            sanityCounter[int(trueClass)] = sanityCounter[int(trueClass)] + 1 #Add one to the count of saved samples for this class
    #Now we have gone through the entire network, make sure we have enough samples
    for c in range(0, numClasses):
        if sanityCounter[c] != numSamplesPerClass:
            raise ValueError("The network does not have enough correctly predicted samples for this class.")
    #Assume we have enough samples now, restore in a properly shaped array 
    #xCorrect = torch.zeros((totalSamplesRequired, xData.shape[1], xData.shape[2], xData.shape[3]))
    xCorrect = torch.zeros(((totalSamplesRequired,) + sampleShape))
    yCorrect = torch.zeros((totalSamplesRequired))
    currentIndex = 0 #indexing for the final array
    for c in range(0, numClasses): #Go through each class
        for j in range(0, numSamplesPerClass): #For each sample in the class store it 
            xCorrect[currentIndex] = correctlyClassifiedSamples[c,j]
            yCorrect[currentIndex] = c
            #yCorrect[currentIndex, c] = 1.0
            currentIndex = currentIndex + 1 
    #return xCorrect, yCorrect
    cleanDataLoader = TensorToDataLoader(xCorrect, yCorrect, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return cleanDataLoader

#Manually shuffle the data loader assuming no transformations
def ManuallyShuffleDataLoader(dataLoader):
    xTest, yTest = DataLoaderToTensor(dataLoader)
    #Shuffle the indicies of the samples 
    indexList = []
    for i in range(0, xTest.shape[0]):
        indexList.append(i)
    shuffle(indexList)
    #Shuffle the samples and put them back in the dataloader 
    xTestShuffle = torch.zeros(xTest.shape)
    yTestShuffle = torch.zeros(yTest.shape)
    for i in range(0, xTest.shape[0]): 
        xTestShuffle[i] = xTest[indexList[i]]
        yTestShuffle[i] = yTest[indexList[i]]
    dataLoaderShuffled = TensorToDataLoader(xTestShuffle, yTestShuffle, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return dataLoaderShuffled

#Takes the Tiny ImageNet main dir (as string) as input, imgSize, batchSize and shuffle (true/false)
#Returns the train loader as output 
def LoadTinyImageNetTrainingData(mainDir, imgSize, batchSize, shuffle):
    tinyImageNetTrainDir = mainDir + "//train"
    if imgSize != 64:
        print("Warning: The default size of Tiny ImageNet is 64x64. You are not using this image size for the dataloader.")
    transform = transforms.Compose([
    # you can add other transformations in this list
    transforms.Resize((imgSize, imgSize)),
    transforms.ToTensor()])
    dataset = datasets.ImageFolder(tinyImageNetTrainDir, transform=transform)
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=shuffle, num_workers=4)
    return trainLoader

#Takes the Tiny ImageNet main dir (as string) as input, imgSize, batchSize and shuffle (true/false)
#Returns the test loader as output 
def LoadTinyImageNetValidationData(mainDir, imgSize, batchSize):
    #Create the dictionary to get the class labels 
    imgNum = 10000 #This part hardcoded for Tiny ImageNet
    wnidsDir = mainDir + "//tiny-imagenet-200//wnids.txt"
    file1 = open(wnidsDir)
    Lines = file1.readlines() 
    classDict = {} #Start a dictionary for the classes 
    classIndex = 0
    for i in range(0, len(Lines)):
        classDict[Lines[i][0:len(Lines[i])-1]] = classIndex
        classIndex = classIndex + 1 
    #Match validation data with the corresponding labels 
    valLabelDir = mainDir + "//val//val_annotations.txt"
    file2 = open(valLabelDir)
    LinesV = file2.readlines() 
    yData = torch.zeros(imgNum, dtype=torch.long) #Without long type cannot train with cross entropy and PyTorch will throw an error
    #Debugging code
    dirTrainList = mainDir + "//tiny-imagenet-200//train//"
    trainClassArrayStrings = os.listdir(dirTrainList)
    for i in range(0, len(LinesV)):
        currentLines = LinesV[i].split("\t")
        classLabelString = currentLines[1]
        sanityChecker = 0
        if i == 9999:
            breaker = 0
        #Go through and match the class string to the index folder in the training data 
        for j in range(0, len(trainClassArrayStrings)):
            if trainClassArrayStrings[j] == classLabelString:
                yData[i] = j #match the class label to the right index in the training data  
                sanityChecker = sanityChecker + 1
        if sanityChecker != 1:
            print("Failed on sample "+str(i))
            print("Sample has class label string:"+ classLabelString)
            raise ValueError("Could not match validation sample with class label in the training set. See above print statements to debug.")
    #Go through and get the class label for each data point 
    #for i in range(0, len(LinesV)):
    #    currentLines = LinesV[i].split("\t")
    #    classLabelString = currentLines[1]
    #    classAsInt = classDict[classLabelString]
    #    yData[i] = classAsInt
    #Load the validation data 
    xData = torch.zeros(imgNum, 3, imgSize, imgSize)
    t = transforms.ToTensor()
    rs = transforms.Resize((imgSize, imgSize))
    valImageDir = mainDir + "//val//images//"
    for i in range(0, imgNum):
        imgName = valImageDir + "val_"+str(i)+".JPEG"
        #currentImage = cv2.imread(imgName)
        currentImage = PIL.Image.open(imgName)
        xData[i] = t(rs(currentImage))
        if i % 1000 == 0:
            print("Loaded up to image:", i)
    #valData = datasets.ImageFolder(root=valImageDir, transform=None)
    #for i in range(0, imgNum):
    #    x = rs(valData[i][0])
    #    xData[i] = t(x)
    #    if i % 1000 == 0:
    #        print("Loaded up to image:", i)
    #debug code
    #ShowImages(xData.numpy().transpose((0,2,3,1)), xData.numpy().transpose((0,2,3,1)))
    #end debug code
    finalLoader = TensorToDataLoader(xData, yData, transforms = None, batchSize = batchSize, randomizer = None)
    return finalLoader










# This is a library of functions we'll use to evaluate VoterLab classifier models
# Many of these functions bear resemblance to those in DataManagerPyTorch since they were modified to work with binary classifiers
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torchsummary import summary
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.optim.lr_scheduler as schedulers
#import seaborn as sns
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models.densenet import DenseNet
import torch.optim as optim
#import AttackWrappersWhiteBoxP as attack
from random import shuffle
#import APGD

import os
from PIL import Image
from random import shuffle
# import sys
# sys.path.insert(0, "/home/aayushi/GitHub/Busting-the-Ballot/")
# import Utilities.DataManagerPytorch as datamanager
# import Utilities.LoadVoterData
# from Utilities.LoadVoterData import LoadData

# Save all loaders to color & greyscale directories
saveDirRGB =  os.path.dirname(os.getcwd())  + "//Train//Trained_RGB_VoterLab_Models//"
if not os.path.exists(saveDirRGB): os.makedirs(saveDirRGB)
saveDirGrayscale = os.path.dirname(os.getcwd())  + "//Train//Trained_Grayscale_VoterLab_Models//"
if not os.path.exists(saveDirGrayscale): os.makedirs(saveDirGrayscale)

# Given a dataloader, return a balanced dataloader with numSamplesRequired // numClasses examples for each class
def ReturnBalancedDataLoader(loader, numClasses, numSamplesRequired, batchSize):
    # Create datasets to store balanced example loader
    #loader = datamanager.ManuallyShuffleDataLoader(loader)
    xData, yData = DataLoaderToTensor(loader)
    sampleShape = GetOutputShape(loader)
    xBal = torch.zeros(((numSamplesRequired, ) + sampleShape))
    yBal = torch.zeros((numSamplesRequired))
    # Manually go through dataset until we get all samples of each class up to numSamplesRequired
    numClassesAdded = [int(numSamplesRequired / numClasses) for i in range(0, numClasses)]
    curIndex = 0
    for i, (data, target) in enumerate(loader):
        for j in range(0, target.size(dim = 0)):
            if numClassesAdded[int(target[j])] > 0:
                xBal[curIndex] = data[j]
                yBal[curIndex] = target[j]
                numClassesAdded[int(target[j])] = numClassesAdded[int(target[j])] - 1
                curIndex += 1
    # Create dataloader, manually shuffle, then return
    loader = TensorToDataLoader(xData = xBal, yData = yBal, batchSize = batchSize)
    #loader = datamanager.ManuallyShuffleDataLoader(loader)
    print(numClassesAdded)
    print("Balanced Loader Shape: ", GetOutputShape(loader))
    return loader


# Return training and validation loader 
def ReturnVoterLabDataLoaders(imgSize, loaderCreated, batchSize, loaderType):
    # Load training and validation sets, normalize from 0-255 to 0-1 datarange, perform greyscale conversion, save
    if not loaderCreated:
        originalBatchSize = batchSize
        # Split examples containing bubbles & no bubbles into train & test loaders (make sure there's no overlap)
        # This allows us to take overlap of validation examples from models exclusively trained on bubbles and those not
        xtrainBubbles, ytrainBubbles, xtestBubbles, ytestBubbles = OnlyBubbles("data/data_Blank_Vote_Questionable.h5")
        xtrainCombined, ytrainCombined, xtestCombined, ytestCombined = LoadRawDataBalanced("data/data_Blank_Vote_Questionable.h5")
        
        # Normalize from 0-255 range to 0-1
        xtrainCombined /= 255
        xtestCombined /= 255
        xtrainBubbles /= 255
        xtestBubbles  /= 255
        batchSize = 64
        print("X Train (Bubbles & No Bubbles) Size (Before No Blacks) = ", xtrainCombined.size())
        print("X Train Only Bubbles Size (Before No Blacks) = ", xtrainBubbles.size())
        
        # Create dataloaders with bubbles & non-bubbles, balance then shuffle them
        count = ReturnNumClasses(yData = ytrainCombined, numClasses = 2)
        print("Count: ", count)
        numSamplesRequired = 0
        if count[0] < count[1]:     numSamplesRequired = count[0]
        else:                       numSamplesRequired = count[1]
        trainLoaderCombined = TensorToDataLoader(xtrainCombined, ytrainCombined, batchSize = batchSize)
        trainLoaderCombined = ManuallyShuffleDataLoader(trainLoaderCombined)
        trainLoaderBalCombined = ReturnBalancedLoader(loader = trainLoaderCombined, numClasses = 2, numSamplesRequired = numSamplesRequired, batchSize = batchSize)
        valLoaderCombined = TensorToDataLoader(xtestCombined, ytestCombined, batchSize = batchSize)
        valLoaderCombined = ManuallyShuffleDataLoader(valLoaderCombined)
        xTrain, yTrain = DataLoaderToTensor(trainLoaderCombined)
        print("Full Train Loader Size (Before Greyscale): ", xTrain.size())

        # Create dataloaders with only bubbles, balance then shuffle them
        count = ReturnNumClasses(yData = ytrainBubbles, numClasses = 2)
        print("Count (Only Bubbles): ", count)
        numSamplesRequired = 0
        if count[0] < count[1]:     numSamplesRequired = count[0]
        else:                       numSamplesRequired = count[1]
        trainLoaderBubbles = TensorToDataLoader(xtrainBubbles, ytrainBubbles, batchSize = batchSize)
        trainLoaderBubbles = ManuallyShuffleDataLoader(trainLoaderBubbles)
        trainLoaderBalBubbles = ReturnBalancedLoader(loader = trainLoaderBubbles, numClasses = 2, numSamplesRequired = numSamplesRequired, batchSize = batchSize)
        valLoaderBubbles = TensorToDataLoader(xtestBubbles, ytestBubbles, batchSize = batchSize)
        xTrain, yTrain = DataLoaderToTensor(trainLoaderBubbles)
        print("Bubble Train Loader Size (Before Greyscale): ", xTrain.size())

        # Perform greyscale conversion on all loaders
        trainLoaderGreyscaleCombined = ConvertToGreyScale(dataLoader = trainLoaderCombined, imgSize = imgSize, batchSize = batchSize)
        trainLoaderGreyscaleBalCombined = ConvertToGreyScale(dataLoader = trainLoaderBalCombined, imgSize = imgSize, batchSize = batchSize)
        valLoaderGreyscaleCombined = ConvertToGreyScale(dataLoader = valLoaderCombined, imgSize = imgSize, batchSize = batchSize)
        trainLoaderGreyscaleBubbles = ConvertToGreyScale(dataLoader = trainLoaderBubbles, imgSize = imgSize, batchSize = batchSize)
        trainLoaderGreyscaleBalBubbles = ConvertToGreyScale(dataLoader = trainLoaderBalBubbles, imgSize = imgSize, batchSize = batchSize)
        valLoaderGreyscaleBubbles = ConvertToGreyScale(dataLoader = valLoaderBubbles, imgSize = imgSize, batchSize = batchSize)
        xData, yData = DataLoaderToTensor(trainLoaderGreyscaleCombined)
        batchSize = originalBatchSize
        print("Train Loader Size (After Greyscale): ", xData.size())

        # Save all loaders to color & greyscale directories
        torch.save({'TrainLoaderCombined': trainLoaderCombined, 'TrainLoaderBalCombined': trainLoaderBalCombined, 'ValLoaderCombined': valLoaderCombined, 'TrainLoaderBubbles': trainLoaderBubbles, 'TrainLoaderBalBubbles': trainLoaderBalBubbles, 'ValLoaderBubbles': valLoaderBubbles}, os.path.join(saveDirRGB, "TrainLoaders.th"))
        torch.save({'TrainLoaderCombined': trainLoaderGreyscaleCombined, 'TrainLoaderBalCombined': trainLoaderGreyscaleBalCombined, 'ValLoaderCombined': valLoaderGreyscaleCombined, 'TrainLoaderBubbles': trainLoaderGreyscaleBubbles, 'TrainLoaderBalBubbles': trainLoaderGreyscaleBalBubbles, 'ValLoaderBubbles': valLoaderGreyscaleBubbles}, os.path.join(saveDirGrayscale, "TrainGrayscaleLoaders.th"))
        torch.save(valLoaderBubbles, os.path.join(saveDirRGB, "ValBubbles.th"))
        torch.save(valLoaderGreyscaleBubbles, os.path.join(saveDirGrayscale, "ValLoaders.th"))
    
    # If dataloaders were already created, load color/greyscale based on imgSize
    else:
        if imgSize[0] == 3:
            checkpoint = torch.load(os.path.dirname(os.getcwd()) + "/Train/Trained_RGB_VoterLab_Models/TrainLoaders.th", map_location = torch.device("cpu"))
            trainLoaderCombined = checkpoint['TrainLoaderCombined']
            trainLoaderBalCombined = checkpoint['TrainLoaderBalCombined']
            valLoaderCombined = checkpoint['ValLoaderCombined']
            trainLoaderBubbles = checkpoint['TrainLoaderBubbles']
            trainLoaderBalBubbles = checkpoint['TrainLoaderBalBubbles']
            valLoaderBubbles = checkpoint['ValLoaderBubbles']
        if imgSize[0] == 1:
            checkpoint = torch.load(os.path.dirname(os.getcwd()) + "/Train/Trained_Grayscale_VoterLab_Models/TrainGrayscaleLoaders.th", map_location = torch.device("cpu"))
            # checkpoint = torch.load(os.getcwd() + "/Train/Trained_Grayscale_VoterLab_Models/TrainGrayscaleLoaders.th", map_location = torch.device("cpu"))
            trainLoaderCombined = checkpoint['TrainLoaderCombined']
            trainLoaderBalCombined = checkpoint['TrainLoaderBalCombined']
            valLoaderCombined = checkpoint['ValLoaderCombined']
            trainLoaderBubbles = checkpoint['TrainLoaderBubbles']
            trainLoaderBalBubbles = checkpoint['TrainLoaderBalBubbles']
            valLoaderBubbles = checkpoint['ValLoaderBubbles']
    
    # Set type of dataloader
    trainLoader = None
    valLoader = None
    if loaderType == 'Bubbles':
        trainLoader = trainLoaderBubbles
        valLoader = valLoaderBubbles
    if loaderType == 'BalBubbles':
        trainLoader = trainLoaderBalBubbles
        valLoader = valLoaderBubbles
    if loaderType == 'Combined':
        trainLoader = trainLoaderCombined
        valLoader = valLoaderCombined
    if loaderType == 'BalCombined':
        trainLoader = trainLoaderBalCombined
        valLoader = valLoaderCombined
        
    # Set dataloader batch sizes
    xTrain, yTrain = DataLoaderToTensor(trainLoader)
    xVal, yVal = DataLoaderToTensor(valLoader)
    trainLoader = TensorToDataLoader(xData = xTrain, yData = yTrain, batchSize = batchSize)
    valLoader = TensorToDataLoader(xData = xVal, yData = yVal, batchSize = batchSize)
    
    # Return dataloaders
    return trainLoader, valLoader


#Manually shuffle the data loader assuming no transformations
def ManuallyShuffleDataLoader(dataLoader):
    xTest, yTest = DataLoaderToTensor(dataLoader)
    #Shuffle the indicies of the samples 
    indexList = []
    for i in range(0, xTest.shape[0]):
        indexList.append(i)
    shuffle(indexList)
    #Shuffle the samples and put them back in the dataloader 
    xTestShuffle = torch.zeros(xTest.shape)
    yTestShuffle = torch.zeros(yTest.shape)
    for i in range(0, xTest.shape[0]): 
        xTestShuffle[i] = xTest[indexList[i]]
        yTestShuffle[i] = yTest[indexList[i]]
    dataLoaderShuffled = TensorToDataLoader(xTestShuffle, yTestShuffle, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return dataLoaderShuffled


# Return separate mark and non-mark dataloaders
def SplitLoader(dataLoader):
    sampleShape = GetOutputShape(dataLoader)
    xData, yData = DataLoaderToTensor(dataLoader)
    voteData = torch.zeros((xData.size(dim=0)//2,) + sampleShape)
    nonvoteData = torch.zeros((xData.size(dim=0)//2,) + sampleShape)
    voteDataIndex, nonvoteDataIndex = 0, 0
    for i, (data, target) in enumerate(dataLoader):
        batchSize = int(data.shape[0])
        for j in range(0, batchSize):
            if int(target[j]) == 0: 
                voteData[voteDataIndex] = data[j]
                voteDataIndex += 1 
            if int(target[j]) == 1:
                nonvoteData[nonvoteDataIndex] = data[j]
                nonvoteDataIndex += 1 
    voteLoader = TensorToDataLoader(xData = voteData, yData = torch.zeros(xData.size(dim=0)//2), batchSize = 64)
    nonvoteLoader = TensorToDataLoader(xData = nonvoteData, yData = torch.ones(xData.size(dim=0)//2), batchSize = 64)
    return voteLoader, nonvoteLoader


# Outputs accuracy given data loader and binary classifier
# validateD function from DataManagerPyTorch doesn't work with binary classification since it outputs argmax
def validateBC(model, loader, device, returnLoaders = False, printAcc = True, returnWhereWrong = False):
    model.eval()
    numCorrect = 0
    batchTracker = 0
    # Without adding to model loss, go through each batch, compute output, tally how many examples our model gets right
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            sampleSize = data.shape[0]
            target = target.unsqueeze(1)
            batchTracker += sampleSize
            data = data.to(device)
            print(data.size())
            output = model(data) #.float()
            for j in range(0, sampleSize):
                if output[j] >= 0.5 and int(target[j]) == 1:   numCorrect += 1
                if output[j] < 0.5  and int(target[j]) == 0:   numCorrect += 1
    # Compute raw accuracy
    acc = numCorrect / float(len(loader.dataset))
    if printAcc:
        print("--------------------------------------")
        print("Accuracy: ", acc)
        print("--------------------------------------")
    # Go through examples again, save them in right/wrong dataloaders to return
    if returnLoaders:
        xData, yData = DataLoaderToTensor(loader)
        if returnWhereWrong: wrongLocation = torch.zeros((len(loader.dataset)))
        #xData, yData = datamanager.DataLoaderToTensor(loader)
        sampleShape = GetOutputShape(loader)
        xRight = torch.zeros(((numCorrect, ) + sampleShape))
        yRight = torch.zeros((numCorrect))
        numWrong = int(len(loader.dataset) - numCorrect)
        xWrong = torch.zeros(((numWrong, ) + sampleShape))
        yWrong = torch.zeros((numWrong))
        loaderWrongTracker = 0
        loaderRightTracker = 0
        loaderTracker = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(loader):
                data = data.to(device)
                # This was saving the sample size from the previous enumerate, so the remaining examples were all zero
                # AKA instead of batch size 64, it was going through less than 64 examples --> THIS is the fix!!!
                batchSize = int(data.shape[0])
                output = model(data)
                for j in range(0, batchSize):
                    if (output[j] >= 0.5 and int(target[j]) == 0) or (output[j] <= 0.5 and int(target[j]) == 1):
                        xWrong[loaderWrongTracker] = data[j]
                        yWrong[loaderWrongTracker] = target[j]
                        loaderWrongTracker += 1
                        if returnWhereWrong: wrongLocation[loaderTracker] = 1
                    else:
                        xRight[loaderRightTracker] = data[j]
                        yRight[loaderRightTracker] = target[j]
                        loaderRightTracker += 1
                    loaderTracker += 1
        wrongLoader = TensorToDataLoader(xData = xWrong, yData = yWrong, batchSize = 64)
        rightLoader = TensorToDataLoader(xData = xRight, yData = yRight, batchSize = 64)
        if returnWhereWrong: return acc, rightLoader, wrongLoader, numCorrect, wrongLocation
        return acc, rightLoader, wrongLoader, numCorrect
    # Return final accuracy
    return acc


# Outputs accuracy given data loader and classifier
# validateD function from DataManagerPyTorch doesn't work with binary classification since it outputs argmax
def validateReturn(model, loader, device, returnLoaders = False, printAcc = True, returnWhereWrong = False):
    model.eval()
    numCorrect = 0
    batchTracker = 0
    # Without adding to model loss, go through each batch, compute output, tally how many examples our model gets right
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            sampleSize = input.shape[0]
            target = target.unsqueeze(1)
            batchTracker += sampleSize
            input = input.to(device)
            output = model(input.to(device)) #.float()
            for j in range(0, sampleSize):
                if output[j].argmax(axis = 0) == int(target[j]):  numCorrect += 1
    # Compute raw accuracy
    acc = numCorrect / float(len(loader.dataset))
    if printAcc:
        print("--------------------------------------")
        print("Accuracy: ", acc)
        print("--------------------------------------")
    # Go through examples again, save them in right/wrong dataloaders to return
    if returnLoaders:
        xData, yData = DataLoaderToTensor(loader)
        if returnWhereWrong: wrongLocation = torch.zeros((len(loader.dataset)))
        #xData, yData = datamanager.DataLoaderToTensor(loader)
        sampleShape = GetOutputShape(loader)
        xRight = torch.zeros(((numCorrect, ) + sampleShape))
        yRight = torch.zeros((numCorrect))
        numWrong = int(len(loader.dataset) - numCorrect)
        xWrong = torch.zeros(((numWrong, ) + sampleShape))
        yWrong = torch.zeros((numWrong))
        loaderWrongTracker = 0
        loaderRightTracker = 0
        loaderTracker = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(loader):
                data = data.to(device)
                # This was saving the sample size from the previous enumerate, so the remaining examples were all zero
                # AKA instead of batch size 64, it was going through less than 64 examples --> THIS is the fix!!!
                batchSize = int(data.shape[0])
                output = model(data)
                for j in range(0, batchSize):
                    if output[j].argmax(axis = 0) != int(target[j]):
                        xWrong[loaderWrongTracker] = data[j]
                        yWrong[loaderWrongTracker] = target[j]
                        loaderWrongTracker += 1
                        if returnWhereWrong: wrongLocation[loaderTracker] = 1
                    else:
                        xRight[loaderRightTracker] = data[j]
                        yRight[loaderRightTracker] = target[j]
                        loaderRightTracker += 1
                    loaderTracker += 1
        wrongLoader = TensorToDataLoader(xData = xWrong, yData = yWrong, batchSize = 64)
        rightLoader = TensorToDataLoader(xData = xRight, yData = yRight, batchSize = 64)
        if returnWhereWrong: return acc, rightLoader, wrongLoader, numCorrect, wrongLocation
        return acc, rightLoader, wrongLoader, numCorrect
    # Return final accuracy
    return acc


# Given a dataloader and 1D boolean tensor, return tensor where boolean tensor is true
def ReturnOnlyLocation(loader, boolTensor, numOfOnes, batchSize):
    xData, yData = DataLoaderToTensor(loader)
    newXData = torch.zeros(((numOfOnes, ) + GetOutputShape(loader)))
    newYData = torch.zeros((numOfOnes))
    dataTracker = 0
    for i in range(len(numOfOnes)):
        if numOfOnes[i].bool():
            newXData[dataTracker] = xData[i]
            newYData[dataTracker] = yData[i]
            dataTracker += 1
    return TensorToDataLoader(xData = newXData, yData = newYData, batchSize = batchSize)


# Given number of classes (2 for binary classification), return how many examples in a dataset's output belong to each class
# We use this to make sure ReturnBalancedLoader works correctly
def ReturnNumClasses(yData, numClasses):
    count = [0 for i in range(0, numClasses)]
    for i in range(yData.size(dim = 0)):
        count[int(yData[i])] += 1
    return count


# Given a dataloader and a number of classes, return a classwise balanced dataloader with numSamplesRequired examples
def ReturnBalancedLoader(loader, numClasses, numSamplesRequired, batchSize):
    # Create datasets to store balanced example loader
    xData, yData = DataLoaderToTensor(loader)
    sampleShape = GetOutputShape(loader)
    xBal = torch.zeros(((numSamplesRequired, ) + sampleShape))
    yBal = torch.zeros((numSamplesRequired))
    # Manually go through dataset until we get all samples of each class up to numSamplesRequired
    numClassesAdded = [int(numSamplesRequired / numClasses) for i in range(0, numClasses)]
    curIndex = 0
    for i, (data, target) in enumerate(loader):
        for j in range(0, target.size(dim = 0)):
            if numClassesAdded[int(target[j])] > 0:
                xBal[curIndex] = data[j]
                yBal[curIndex] = target[j]
                numClassesAdded[int(target[j])] = numClassesAdded[int(target[j])] - 1
                curIndex += 1
    '''
    for i in range(0, yData.size(dim = 0)):
        if numClassesAdded[int(yData[i])] > 0:
            xBal[curIndex] = xData[i]
            yBal[curIndex] = yData[i]
            numClassesAdded[int(yData[i])] = numClassesAdded[int(yData[i])] - 1
            curIndex += 1
    '''
    # Create dataloader, manually shuffle, then return
    loader = TensorToDataLoader(xData = xBal, yData = yBal, batchSize = batchSize)
    #loader = ManuallyShuffleDataLoader(loader)
    print("Balanced Loader Shape: ", GetOutputShape(loader))
    return loader


# Given a dataloader, return a balanced loader of at most totalSamplesRequired / numClasses examples of each class which model classifies correctly 
# This is ONLY for binary classifiers where the output is one unit
def GetCorrectlyIdentifiedSamplesBalanced(device, model, batchSize, totalSamplesRequired, dataLoader, numClasses):
    xData, yData = DataLoaderToTensor(dataLoader)
    xCorrectBal = torch.zeros(((totalSamplesRequired, ) + GetOutputShape(dataLoader)))
    yCorrectBal = torch.zeros((totalSamplesRequired))
    # Compute output for each batch, store totalSamplesRequired/numClasses examples for each class
    samplesForEachClass = int(totalSamplesRequired / numClasses)
    # print(samplesForEachClass)
    examplesForEachClass = [0 for i in range(0, numClasses)]
    correctBalIndex = 0
    for i, (data, target) in enumerate(dataLoader):
        output = model(data.to(device))
        for j in range(0, target.size(dim = 0)):
            if (output[j] >= 0.5 and int(target[j]) == 1) or (output[j] < 0.5 and int(target[j]) == 0):
                if (examplesForEachClass[int(target[j])] < samplesForEachClass):
                    xCorrectBal[correctBalIndex] = data[j]
                    yCorrectBal[correctBalIndex] = target[j]
                    correctBalIndex += 1
                    examplesForEachClass[int(target[j])] += 1
    # Zip into dataloader, shuffle, return
    correctBalLoader = TensorToDataLoader(xData = xCorrectBal, yData = yCorrectBal, batchSize = batchSize)
    #correctBalLoader = datamanager.ManuallyShuffleDataLoader(correctBalLoader)
    return correctBalLoader


# Display examples from each class and corresponding adversarial example
def DisplayNumValAndAdvExamples (numExamples, valLoader, advLoader, numClasses, classNames, numSamples, model, batchNum, saveTag, greyScale, addText):    
    # .transpose(...) fixes TypeError: Invalid shape (3, 32, 32) for image data
    xValT, yValT = DataLoaderToTensor(valLoader)
    xAdvT, yAdvT = DataLoaderToTensor(advLoader)
    
    # Transpose x-data in numpy 0 - 1 range 
    xVal = xValT.detach().numpy().transpose((0,2,3,1))
    xAdv = xAdvT.detach().numpy().transpose((0,2,3,1))
    yVal = yValT.numpy()
    
    # Find five examples from each class from validation and adv examples
    remainingClasses = [numExamples for i in range(numClasses)]
    xValExamples = dict()
    xValAccuracies = dict()
    xAdvExamples = dict()
    xAdvAccuracies = dict()
    
    for j in range(numClasses):
        for i in range(numSamples):
            #print(yVal[i].item())
            if yVal[i].item() >= 0.5: curY = 1
            else:                     curY = 0
            #curY = int(yVal[i].item())
            if curY == j:
                if remainingClasses[curY] > 0:
                    xValExamples[classNames[curY] + str(remainingClasses[curY])] = xVal[i]
                    if addText:
                        curPrediction = (model(xValT[i].unsqueeze(0).cuda())[0]).float().cpu().detach().numpy()
                        if ((curY == 1) and (curPrediction.item() <= 0.5)) or ((curY == 0) and (curPrediction.item() >= 0.5)):
                            xValAccuracies[classNames[curY] + str(remainingClasses[curY])] = "Misclassified"
                        else:
                            xValAccuracies[classNames[curY] + str(remainingClasses[curY])] = "Correct"
                        #xValAccuracies[classNames[curY] + str(remainingClasses[curY])] = xValAccuracies[classNames[curY]].float()
                        #xValAccuracies[classNames[curY] + str(remainingClasses[curY])] = xValAccuracies[classNames[curY]].detach().numpy()
                        # xValAccuracies[classNames[curY]] = xValAccuracies[classNames[curY]][0, curY]
            
                    xAdvExamples[classNames[curY] + str(remainingClasses[curY])] = xAdv[i]
                    if addText:
                        advOutput = model(xAdvT[i].unsqueeze(0).cuda())[0]
                        advOutput = advOutput.float().cpu().detach().numpy()
                        #most_confidence_class = int(adv_output.argmax())
                        #xAdvClasses.append(most_confidence_class)
                        if ((curY == 1) and (advOutput.item() <= 0.5)) or ((curY == 0) and (advOutput.item() >= 0.5)):
                            xAdvAccuracies[classNames[curY] + str(remainingClasses[curY])] = "Misclassified" #adv_output
                        else:
                            xAdvAccuracies[classNames[curY] + str(remainingClasses[curY])] = "Correct"
                        #xAdvAccuracies[classNames[curY]] = xAdvAccuracies[most_confidence_class][0, most_confidence_class]
                    '''
                    x_adv_balanced_accuracies[class_names[cur_y]] = model(x_adv_t[i].unsqueeze(0))
                    x_adv_balanced_accuracies[class_names[cur_y]] = x_adv_balanced_accuracies[class_names[cur_y]].float()
                    x_adv_balanced_accuracies[class_names[cur_y]] = x_adv_balanced_accuracies[class_names[cur_y]].detach().numpy()
                    x_adv_balanced_accuracies[class_names[cur_y]] = x_adv_balanced_accuracies[class_names[cur_y]][0, cur_y]
                    '''
            
                    remainingClasses[curY] = remainingClasses[curY] - 1
            
                #if remainingClasses == [0 for i in range(numClasses)]:
                #    break
    
    #Show 20 images, 10 in first and row and 10 in second row 
    if greyScale: plt.gray()
    n = numClasses * numExamples  # how many images we will display
    plt.figure(figsize=(numExamples * 2, 6))   
    for i in range(numClasses):    
        for j in range(numExamples):
            # display original
            ax = plt.subplot(2, n, numExamples * i + (j+1))
            plt.imshow(xValExamples[classNames[i] + str(j+1)])
            #plt.text(.01, .5, class_names[i] + ': ' + str(x_val_balanced_accuracies[class_names[i]]), ha='center', fontsize = 'xx-small')
            if addText: 
                ax.set_xlabel(xValAccuracies[classNames[i] + str(j+1)], fontsize = 'x-small')
                #plt.text(.01, .5, classNames[i] + str(j+1) + ': ' + str(xValAccuracies[classNames[i] + str(j+1)]), fontsize = 'x-small')
            #ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
            # display reconstruction
            ax = plt.subplot(2, n, numExamples * i + (j+1) + n) 
            plt.imshow(xAdvExamples[classNames[i] + str(j+1)])
            if addText: # classNames[i] + str(j+1) + ': ' + 
                ax.set_xlabel(xAdvAccuracies[classNames[i] + str(j+1)], fontsize = 'x-small')
                #plt.text(.01, .5, classNames[i] + str(j+1) + ': ' + str(xAdvAccuracies[classNames[i] + str(j+1)]), fontsize = 'x-small')
            #y
            #ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
    plt.show()
    plt.savefig(saveTag)
    plt.close()


# Convert each example in dataloader to one-channel greyscale
def ConvertToGreyScale (dataLoader, imgSize, batchSize):
    # Convert xData into numpy array, create empty dataset for greyscale images
    xData, yData = DataLoaderToTensor(dataLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    newXData = torch.zeros(xData.size(dim = 0), 1, imgSize[1], imgSize[2])

    # Manually take average of each channel in each image
    # Matlab greyscale conversion formula used: 0.2989 * R + 0.5870 * G + 0.1140 * B
    for i in range(xData.size(dim = 0)):
        newXData[i] = 0.2989 * xData[i][0] + 0.5870 * xData[i][1] + 0.1140 * xData[i][2]

    # Return transformed dataloader
    print("Greyscale X-Data Size: ", newXData.size())
    return TensorToDataLoader(xData = newXData, yData = yData, randomizer = None, batchSize = batchSize)


# Given two models, get the first correctly overlapping balanced samples
def GetFirstCorrectlyOverlappingSamplesBalanced(device, imgSize, sampleNum, batchSize, numClasses, dataLoader, modelA, modelB):
    xData, yData = DataLoaderToTensor(dataLoader)
    # Get accuracy array from each model
    accArrayA = validateDA(valLoader = dataLoader, model = modelA, device = device)
    accArrayB = validateDA(valLoader = dataLoader, model = modelB, device = device)
    accArray = accArrayA + accArrayB
    # Create datasets to store overlapping examples, manually go through each example
    xClean = torch.zeros(sampleNum, imgSize[0], imgSize[1], imgSize[2])
    yClean = torch.zeros(sampleNum)
    sampleIndexer = 0
    numSamplesPerClass = [0 for i in range(0, numClasses)]
    for i in range(0, xData.size(dim = 0)):
        currentClass = int(yData[i])
        if accArray[i] == 2.0 and numSamplesPerClass[currentClass] < int(sampleNum / numClasses):
            xClean[sampleIndexer] = xData[i]
            yClean[sampleIndexer] = yData[i]
            sampleIndexer += 1 
            numSamplesPerClass[currentClass] += 1
    # Return clean data loader
    return TensorToDataLoader(xData = xClean, yData = yClean, batchSize = batchSize)


# Given a dataloader and its adversarial loader, take difference, average, then create heatmap plot
def CreateHeatMap(valLoader, advLoader, greyScale, saveDir, saveName):
    # Get datasets, turn them into numpy arrays
    #xValT, yValT = datamanager.DataLoaderToTensor(valLoader)
    #xAdvT, yAdvT = datamanager.DataLoaderToTensor(advLoader)
    imgSize = ((1, 40, 50) if greyScale else (3, 40, 50))
    #xVal = xValT.detach().numpy().transpose((1, 2, 0))
    #xAdv = xAdvT.detach().numpy().transpose((1, 2, 0))
    advIterator = iter(advLoader)
    
    # Iterate through each example, take difference between val & adv, add it to heatMap
    heatMapVote = np.array([imgSize[1], imgSize[2], imgSize[0]]).astype(dtype='float64')
    heatMapNonVote = np.array([imgSize[1], imgSize[2], imgSize[0]]).astype(dtype='float64')
    
    indexerVote = 0
    indexerNonVote = 0
    for i, (dataVal, targetVal) in enumerate(valLoader):
        dataAdv, targetAdv = next(advIterator)
        for j in range(0, targetVal.size(dim = 0)):
            #if greyScale: plt.gray()
            xVal = dataVal[j].detach().numpy().transpose((1, 2, 0))
            xAdv = dataAdv[j].detach().numpy().transpose((1, 2, 0))
            exampleDiff = np.subtract(xAdv, xVal)
            if int(targetVal[0]) == 0: 
                heatMapVote = np.add(heatMapVote, exampleDiff)
                indexerVote += 1
            if int(targetVal[1]) == 1: 
                heatMapNonVote = np.add(heatMapNonVote, exampleDiff)
                indexerNonVote += 1
            
    #for i in range(len(xVal)): heatMap += (xAdv[i] - xVal[i])
    heatMapVote /= indexerVote
    heatMapNonVote /= indexerNonVote
    
    # Create heatmap plot for EACH channel, save it to specified directory
    if imgSize[0] == 1:
        fig, ax = plt.subplots()
        ax = sns.heatmap(heatMapVote[:, :, 0], linewidth=0.5)
        ax.set_title("Vote_Grayscale " + saveName)
        plt.show()
        plt.savefig(saveDir + "/" + "Vote_Grayscale_" + saveName + ".png")
        plt.close()
        
        fig, ax = plt.subplots()
        ax = sns.heatmap(heatMapNonVote[:, :, 0], linewidth=0.5)
        ax.set_title("Non_Vote_Grayscale " + saveName)
        plt.show()
        plt.savefig(saveDir + "/" + "Non_Vote_Grayscale_" + saveName + ".png")
        plt.close()
    if imgSize[0] == 3:
        colors = ['Red', 'Blue', 'Green']
        for i in range(len(colors)):
            fig, ax = plt.subplots()
            ax = sns.heatmap(heatMapVote[:, :, i], linewidth=0.5)
            ax.set_title("Vote " + colors[i] + " Channel " + saveName)
            plt.show()
            plt.savefig(saveDir + "/" + "Vote " + colors[i] + "_Channel_" + saveName + ".png")
            plt.close()
            
            fig, ax = plt.subplots()
            ax = sns.heatmap(heatMapNonVote[:, :, i], linewidth=0.5)
            ax.set_title("Non-Vote " + colors[i] + " Channel " + saveName)
            plt.show()
            plt.savefig(saveDir + "/" + "Non-Vote " + colors[i] + "_Channel_" + saveName + ".png")
            plt.close()

            
# Given data loader, create a display for each image in given folder
def DisplayImgs (dataLoader, greyScale, saveDir, printMisclassified = False, wrongLocation = None, printRealLabel = True):
    # Enumerate through each example, create plots
    outputShape = GetOutputShape(dataLoader)
    indexer = 0
    for i, (data, target) in enumerate(dataLoader):
        #print("Target Size: ", target.size(dim = 0))
        for j in range(0, target.size(dim = 0)):
            #if greyScale: plt.gray()
            fig, ax = plt.subplots()
            batchSize = target.size(dim = 0)
            xVal = data[j].detach().numpy().transpose((1, 2, 0))
            yVal = target[j].numpy()
            yVal = ('Non-Vote' if yVal > 0.5 else 'Vote')
            #ax.set_title(f'{i}th Batch {j}th Example:')
            #if printRealLabel: ax.set_xlabel(f"Real Label: {yVal}")
            #ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)
            #ax.axis('off')
            '''
            plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            
            #plt.imshow(xVal)
            #plt.imsave(xVal)
            #plt.show()
            '''
            
            # Create save location & name
            saveName = None
            if printMisclassified:
                if int(wrongLocation[indexer]) == 1:
                    saveName = saveDir + "/"  + str(i) + "th Batch " + str(j) + "th Example" + "__" + "Misclassified_" + yVal + ".png"
                    #plt.savefig(saveDir + "/"  + str(i) + "th Batch " + str(j) + "th Example" + "__" + "Misclassified_" + yVal + ".png", 
                    # For no whitebox padding, when plt.savefig do ,bbox_inches = 'tight', pad_inches=0)
                else:
                    saveName = saveDir + "/"  + str(i) + "th Batch " + str(j) + "th Example" + "__" + "Correct_" + yVal + ".png"
                    #plt.savefig(saveDir + "/"  + str(i) + "th Batch " + str(j) + "th Example" + "__" + "Correct_" + yVal + ".png", bbox_inches = 'tight', pad_inches=0)
            else:
                saveName = saveDir + "/" + "__" + str(i) + "th Batch " + str(j) + "th Example__" + yVal + ".png"
                #plt.savefig(saveDir + "/" + "__" + str(i) + "th Batch " + str(j) + "th Example__" + yVal + ".png", bbox_inches = 'tight', pad_inches=0)
            
            # If greyscale, we extend array into 3 channels (to represent 3 color channels, we set them all to the same value)
            extendedXVal = None
            if greyScale:   
                #plt.imsave(fname = saveName, arr = xVal)
                #xVal = (255 * xVal).astype(np.uint32)
                #img = Image.fromarray(xVal, mode = 'RGB')
                extendedXVal = np.zeros((outputShape[1], outputShape[2], 3))
                for k in range(3): extendedXVal[:, :, k] = xVal[:, :, 0]
                plt.imsave(fname = saveName, arr = extendedXVal)
            else:   
                #xVal = (255 * xVal).astype(np.uint8)
                #img = Image.fromarray(xVal)
                plt.imsave(fname = saveName, arr = xVal)
            #img.save(saveName)
            plt.close()
            indexer += 1
    print(str(indexer) + " total images added!")


# Given a data loader and a list of tuples (batch index, index in batch), save these specific images and labels in dataloader
def FindImgs (dataLoader, examplesList, imgSize):
    # Create datasets to store overlapping examples, manually go through each example
    xData = torch.zeros(len(examplesList), imgSize[0], imgSize[1], imgSize[2])
    yData = torch.zeros(len(examplesList))
    # Enumerate through each example, create plots
    curIndex = 0
    for i, (data, output) in enumerate(dataLoader):
        for j in range(data.size(dim = 0)):
            if (i, j) in examplesList:
                xData[curIndex] = data[j]
                yData[curIndex] = output[j]
                curIndex += 1
    # Return dataloader, batch size = 1
    return TensorToDataLoader(xData = xData, yData = yData, batchSize = 1)


# Given a list of models and a dataloader, return two dataloaders of examples all models get correct and wrong
def GetAllRightAndWrongExamples (device, models, numExamples, batchSize, dataLoader, imgSize):
    # Create datasets to store overlapping examples
    xDataRight = torch.zeros(numExamples, imgSize[0], imgSize[1], imgSize[2])
    xDataWrong = torch.zeros(numExamples, imgSize[0], imgSize[1], imgSize[2])
    yDataRight = torch.zeros(numExamples)
    yDataWrong = torch.zeros(numExamples)
    # Go through each example, tally how many total models get this example correct/wrong
    xData, yData = DataLoaderToTensor(dataLoader)
    modelCount = [0 for i in range(xData.size(dim = 0))]
    currentCount = 0
    xData, yData = xData.to(device), yData.to(device)
    for i, (data, output) in enumerate(dataLoader):
        prevCount = currentCount
        for model in models:
            output = model(data.to(device))
            for j in range(0, target.size(dim = 0)):
                if (output[j] >= 0.5 and int(target[j]) == 1) or (output[j] < 0.5 and int(target[j]) == 0):
                    modelCount[currentCount] += 1
                else:
                    modelCount[currentCount] -= 1
                currentCount += 1
            # Reset back for next model
            currentCount = prevCount
    # Go through tally, add examples which all models got correctly/incorrectly to empty datasets
    rightCount = 0
    wrongCount = 0
    for i in range(xData.size(dim = 0)):
        if modelCount[i] == len(models) and rightCount < numExamples: 
            xDataRight[rightCount] = xData[i]
            yDataRight[rightCount] = yData[i]
            rightCount += 1
        if modelCount[i] == - len(models) and wrongCount < numExamples:
            xDataWrong[wrongCount] = xData[i]
            yDataWrong[wrongCount] = yData[i]
            wrongCount += 1
    # Return dataloaders
    rightLoader = TensorToDataLoader(xData = xDataRight, yData = yDataRight, batchSize = batchSize)
    wrongLoader = TensorToDataLoader(xData = xDataWrong, yData = yDataWrong, batchSize = batchSize)
    return rightLoader, wrongLoader


# if __name__ == '__main__':
#     # Generate all training loaders...
#     print(os.path.dirname(os.getcwd()))
#     print("___Creating Training and Validation Loaders...___")
#     ReturnVoterLabDataLoaders(imgSize = (1, 40, 50), loaderCreated = False, batchSize = 64, loaderType = 'BalCombined')


import torch 
import numpy as np
import h5py
import urllib.request
import os

class Progress:
    def __init__(self):
        self.old_percent = 0

    def download_progress_hook(self, count, blockSize, totalSize):
        percent = int(count * blockSize * 100 / totalSize)
        if percent > self.old_percent:
            self.old_percent = percent
            print(percent, '%', end = "\r")
        if percent == 100:
            print()
            print('done!')

def LoadColorData(file):
    #array of dim 2x3x108x3
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    #array of dim 2x3x108
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]

    f = h5py.File(file, "r")


    batches = np.arange((108))
    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    for x in range(len(dset_type)-1):
        for y in range(len(batches)):
                for k in range(len(inf_color_model[0][x][y][0])):
                    small_x = []
                    mean = []
                    var = []
                    img = image_data[0][x][y][k]
                    img = img[:,:,::-1]

#                     if(within_var(img)):
#                         continue

                    #img = img[10:30,15:35,:]

                    small_x.append(inf_color_model[0][x][y][0][k])
                    small_x.append(inf_color_model[0][x][y][1][k])
                    small_x.append(inf_color_model[0][x][y][2][k])
                    small_x.extend(np.mean(img,axis = (0,1)))

                    X.append(small_x)
                    color = colorDef(inf_color_model[0][x][y][0][k],inf_color_model[0][x][y][1][k],inf_color_model[0][x][y][2][k])
                    Y.append(color)
    return X,Y

def GetClasswiseBalanced(y, split, nclasses):
    nsamp = torch.tensor([int(torch.sum(y == i)*split) for i in range(nclasses)])

    classCount = torch.zeros(nclasses)
    indexer = torch.zeros(len(y))
    total = 0
    max = torch.sum(nsamp)

    for i in range(len(y)):
        label = int(y[i])
        if classCount[label] < nsamp[label]:
            indexer[i] = 1
            classCount[label] += 1
            total += 1
        if total >= max:
            break

    return indexer.bool()

def GetNClasswiseBalanced(y, n, nclasses):
    nsamp = torch.tensor([n for i in range(nclasses)])

    classCount = torch.zeros(nclasses)
    indexer = []#torch.zeros(len(y))
    total = 0
    max = torch.sum(nsamp)

    for i in reversed(range(len(y))):
        label = int(y[i])
        if classCount[label] < nsamp[label]:
            indexer.append(i)
            classCount[label] += 1
            total += 1
        if total >= max:
            break

    return torch.tensor(indexer).long()

def colorDef(r,b,g):
    r = int(r)
    b = int(b)
    g = int(g)
    
    #Blue
    if(r <181 and r> 174):
        return 0
    
    #Green
    if(r <190 and r> 180):
        return 1
    
    #White
    if(r > 250 and b > 250 and g > 250 ):
        return 2
    
    #Yellow
    if(r > 200 and b > 200 and g < 200 ):
        return 3
    
    #Pink
    if(r > 200 and b < 160 and b > 140 ):
        return 4
    
    #Salmon
    if(r > 200 and b < 180 and b > 160 ):
        return 5

def collapse(img,b_r,b_g,b_b):
    collapsed_img = np.zeros(shape = (40,50) ,dtype = np.float32)

    img[:,:,0] = img[:,:,0] * (0.02383815) + (b_r*(-0.01898671))
    img[:,:,1] = img[:,:,1] * (0.00010994) + (b_g*(-0.001739))
    img[:,:,2] = img[:,:,2] * (0.00178155) + (b_b*(-0.00044142))

    collapsed_img = np.sum(img,axis = 2)
    
    return collapsed_img.flatten()

def LoadPositionalData(file):
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")


    batches = np.arange((108))
    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    positional_image_data = []
    positional_ground_truth = []
    for x in range(len(dset_type)-1):
        for y in range(len(batches)):
                for k in range(len(inf_color_model[1][x][y][0])):
                    small_x = []

                    img = image_data[1][x][y][k]
                    img = img[:,:,::-1]

                    b_r = inf_color_model[1][x][y][0][k]
                    b_g = inf_color_model[1][x][y][1][k]
                    b_b = inf_color_model[1][x][y][2][k]

                    img = collapse(img,b_r,b_g,b_b)
                    positional_image_data.append(img)
                    positional_ground_truth.append(x)

    return positional_image_data,positional_ground_truth

def LoadBatchSamples(file):

    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")

    batches = np.arange((108))
    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    images = torch.zeros(size = [26*len(batches), 3, 40, 50])
    yvals = torch.zeros(26*len(batches))
    pos = 0
    for y in range(len(batches)):
        for x in range(len(dset_type)-1):
            for k in range(13):
                if k >= len(image_data[1][x][y]):
                    images[pos] = torch.zeros(size = [3,40,50])
                else:
                    img = image_data[1][x][y][k]
                    images[pos] = torch.tensor(img).permute(2, 0, 1)
                yvals[pos] = x
                pos += 1
    return images, yvals, len(batches)

def LoadRawData(file):
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")


    batches = np.arange((108))
    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    pos = 0
    images = torch.zeros(size = [582927, 3, 40, 50])
    yvals = torch.zeros(582927)
    for x in range(len(dset_type)-1):
        for y in range(len(batches)):
                for k in range(len(inf_color_model[1][x][y][0])):
                    img = image_data[1][x][y][k]

                    images[pos] = torch.tensor(img).permute(2, 0, 1)
                    yvals[pos] = x
                    pos += 1

    return images, yvals

# No bubble images
def NoBubbles(file):
    batches = np.arange((108))
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")

    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    pos = 0
    trainx = torch.zeros(size = [600000, 3, 40, 50])
    trainy = torch.zeros(600000)

    pos2 = 0
    testx = torch.zeros(size = [150000, 3, 40, 50])
    testy = torch.zeros(150000)
    # Consider every set besides those which contain bubbles below
    batches = [39, 40, 41, 42, 43, 44, 45, 55, 82, 85, 91, 92, 93, 94, 95, 97, 101, 102, 104, 105]
    noBubbleBatches = []
    for i in range(108): 
        if i not in batches: noBubbleBatches.append(i)
    for x in range(len(dset_type)-1):
        for y in noBubbleBatches:
                total = len(inf_color_model[1][x][y][0])
                for k in range(total):
                    img = image_data[1][x][y][k]
                    if k <= total*.8:
                        trainx[pos] = torch.tensor(img).permute(2, 0, 1)
                        trainy[pos] = x
                        pos += 1
                    else:
                        testx[pos2] = torch.tensor(img).permute(2, 0, 1)
                        testy[pos2] = x
                        pos2 += 1

    return trainx[:pos], trainy[:pos], testx[:pos2], testy[:pos2]
 
#Returns dataset with only bubbles (filled and unfilled)
#You can filter by class using the label vector
def OnlyBubbles(file):
    batches = [39, 40, 41, 42, 43, 44, 45, 55, 82, 85, 91, 92, 93, 94, 95, 97, 101, 102, 104, 105]
    batches = np.arange((108))
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")

    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    pos = 0
    trainx = torch.zeros(size = [470000, 3, 40, 50])
    trainy = torch.zeros(470000)

    pos2 = 0
    testx = torch.zeros(size = [120000, 3, 40, 50])
    testy = torch.zeros(120000)
    batches = [39, 40, 41, 42, 43, 44, 45, 55, 82, 85, 91, 92, 93, 94, 95, 97, 101, 102, 104, 105]
    for x in range(len(dset_type)-1):
        for y in batches:
                total = len(inf_color_model[1][x][y][0])
                for k in range(total):
                    img = image_data[1][x][y][k]
                    if k <= total*.8:
                        trainx[pos] = torch.tensor(img).permute(2, 0, 1)
                        trainy[pos] = x
                        pos += 1
                    else:
                        testx[pos2] = torch.tensor(img).permute(2, 0, 1)
                        testy[pos2] = x
                        pos2 += 1

    return trainx[:pos], trainy[:pos], testx[:pos2], testy[:pos2]

def LoadRawDataBalanced(file):
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")


    batches = np.arange((108))
    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    pos = 0
    trainx = torch.zeros(size = [470000, 3, 40, 50])
    trainy = torch.zeros(470000)

    pos2 = 0
    testx = torch.zeros(size = [120000, 3, 40, 50])
    testy = torch.zeros(120000)
    for x in range(len(dset_type)-1):
        for y in range(len(batches)):
                total = len(inf_color_model[1][x][y][0])
                for k in range(total):
                    img = image_data[1][x][y][k]
                    if k <= total*.8:
                        trainx[pos] = torch.tensor(img).permute(2, 0, 1)
                        trainy[pos] = x
                        pos += 1
                    else:
                        testx[pos2] = torch.tensor(img).permute(2, 0, 1)
                        testy[pos2] = x
                        pos2 += 1

    return trainx[:pos], trainy[:pos], testx[:pos2], testy[:pos2]

#Loads all the data stored in the "position data" part of the .h5 file, and saves the raw images in data/VoterData.torch.
#Data is stored in a dictionary as follows: {"train": {"x": xtrain, "y": ytrain}, "test": {"x": xtest, "y": ytest}}
#Data is split classwise 80/20 train.test. Classwise meaning the training set has 80% of all the "filled" and 80% of all the "empty" inputs.
def SetUpDataset(file, balanced = True):

    if not os.path.isfile(file):
        print ('Downloading Datafile')
        progress = Progress()
        urllib.request.urlretrieve("http://puf-data.engr.uconn.edu/data/data_Blank_Vote_Questionable.h5", file, reporthook=progress.download_progress_hook)

    if not balanced:
        x,y = LoadRawData(file)

        i = GetClasswiseBalanced(y, .2, 2)
        ni = (1-i.int()).bool()

        xtrain = x[ni]
        ytrain = y[ni]

        xtest = x[i]
        ytest = y[i]
        name = "data/VoterData.torch"
    else:
        xtrain,ytrain,xtest,ytest = LoadRawDataBalanced(file)
        name = "data/VoterDataBalanced.torch"
        # name = "/Users/aayushi.verma/Documents/GitHub/Busting-The-Ballot/data/VoterDataBalanced.torch"

    print(xtrain.size(), xtest.size())
    print(ytrain.size(), ytest.size())

    torch.save({"train": {"x": xtrain, "y": ytrain}, "test": {"x": xtest, "y": ytest}}, name)


def LoadData(balanced = True):
    if not balanced:
        name = os.getcwd() + "//data//VoterData.torch"
        # name = "/Users/aayushi.verma/Documents/GitHub/Busting-The-Ballot/data/VoterData.torch"
    else:
        name = os.getcwd() + "//data//VoterDataBalanced.torch"
        # name = "/Users/aayushi.verma/Documents/GitHub/Busting-The-Ballot/data/VoterDataBalanced.torch"
    data = torch.load(name)
    xtrain = data["train"]["x"]
    ytrain = data["train"]["y"]

    xtest = data["test"]["x"]
    ytest = data["test"]["y"]

    return xtrain, ytrain, xtest, ytest


# if __name__ == "__main__":

#     #Data is stored in the /data folder
#     if not os.path.isdir("/Users/aayushi.verma/Documents/GitHub/Busting-The-Ballot/data"):
#         os.mkdir("/Users/aayushi.verma/Documents/GitHub/Busting-The-Ballot/data")

#     file = "/Users/aayushi.verma/Documents/GitHub/Busting-The-Ballot/data/data_Blank_Vote_Questionable.h5"
#     SetUpDataset(file)