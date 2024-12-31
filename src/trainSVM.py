# importing standard packages
import torch
from sklearn.svm import LinearSVC
import os
from matplotlib import pyplot as plt

# importing custom modules
import utils 
import DataManagerPytorch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)
if (torch.cuda.is_available()):
    print('Number of CUDA Devices:', torch.cuda.device_count())
    print('CUDA Device Name:',torch.cuda.get_device_name(0))
    print('CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

# Create folders for trained models 
# base assumption is we are in the directory this script is located in
saveDirRGB =  os.path.dirname(os.getcwd()) + "//Trained_RGB_VoterLab_Models//"
if not os.path.exists(saveDirRGB): os.makedirs(saveDirRGB)
saveDirGrayscale = os.path.dirname(os.getcwd()) + "//Trained_Grayscale_VoterLab_Models//"
if not os.path.exists(saveDirGrayscale): os.makedirs(saveDirGrayscale)


def TrainSVM(useGrayscale=True, combined=False):
    """Function to train the SVM, based on user-specified settings.

    Args:
        useGrayscale (bool, optional): Whether or not to use grayscale dataset. Defaults to True. If False, will use RGB.
        combined (bool, optional): Whether or not to use Combined dataset. Defaults to False. If True, will use Bubbles dataset.
    """

    if useGrayscale:
        dataset_color_tag = 'Gray'
        imgSize = (1, 40, 50)
    else:
        dataset_color_tag = 'RGB'
        imgSize = (3, 40, 50)
    
    if combined:
        dataset_tag = 'Combined'
        loaderType = 'BalCombined'
    else:
        dataset_tag = 'Bubbles'
        loaderType = 'BalBubbles'

    # Hyperparameters
    batchSize = 1
    print("------------------------------------")
    # Get dataloaders
    trainLoader, valLoader = utils.ReturnVoterLabDataLoaders(imgSize = imgSize, loaderCreated = True, batchSize = batchSize, loaderType = loaderType)
    xtrain, ytrain =  utils.DataLoaderToTensor(trainLoader)
    xtest, ytest = utils.DataLoaderToTensor(valLoader)
    xtrain = torch.flatten(xtrain, start_dim = 1)
    xtest = torch.flatten(xtest, start_dim = 1)
    # Normalize
    xtrain /= 255
    xtest /= 255
    # Initialize model and train
    model = pseudoSVM(insize=xtrain.size()[1], outsize=1)
    model.TrainModel(xtrain, ytrain, xtest, ytest)
    # Compute gradients
    xtrain.requires_grad = True
    outputs = model(xtrain)
    loss = torch.mean((outputs - ytrain.float()) ** 2)  # Mean squared error loss
    loss.backward()

    gradients = xtrain.grad  # Get gradients
    zero_gradients = gradients == 0  # Identify zero gradients
    # print(zero_gradients)

    # Create heatmaps
    utils.gradients_heatmap(
        data=gradients, 
        img_size=imgSize, 
        title=f"Gradient Heatmap for {dataset_color_tag} {dataset_tag} SVM", 
        filename=f"{dataset_color_tag}_{dataset_tag}SVM_gradient_heatmap.png", 
        cmap="viridis"
    )

    # Save trained SVM
    saveTag = f"{dataset_color_tag}-SVM-{dataset_tag}"
    saveDir = (saveDirGrayscale if useGrayscale else saveDirRGB)
    torch.save(model.state_dict(), os.path.join(saveDir, saveTag + '.pth'))

class pseudoSVM(torch.nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        self.layer = torch.nn.Linear(insize, outsize, bias=True)
        self.sigmoid = False
        self.s = torch.nn.Sigmoid()

    def forward(self, x):
        if self.sigmoid:
            return self.s(self.layer(x)).T[0]
        return self.layer(x).T[0]
    
    def TrainModel(self, x, y, xt, yt):
        clf = LinearSVC(random_state=0, max_iter=10000, dual=False, C=1e-8, tol=1e-8, penalty='l2', class_weight='balanced', intercept_scaling=1000)
        clf.fit(x.numpy(), y.numpy())
        print("SVM clf score: ", clf.score(xt.numpy(), yt.numpy()))

        with torch.no_grad():
            self.layer.weight = torch.nn.Parameter(torch.tensor(clf.coef_).float())
            self.layer.bias = torch.nn.Parameter(torch.tensor(clf.intercept_).float())
            print(f"The bias is: {self.layer.bias}")

# to train Bubble SVM, uncomment this line:
TrainSVM(useGrayscale=True, combined=False)
TrainSVM(useGrayscale=False, combined=False)
TrainSVM(useGrayscale=True, combined=True)
TrainSVM(useGrayscale=False, combined=True)
