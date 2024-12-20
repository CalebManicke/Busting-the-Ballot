# importing standard packages
import torch
from sklearn.svm import LinearSVC
import os
from matplotlib import pyplot as plt
import seaborn as sns

# importing custom modules
# import Utilities.VoterLab_Classifier_Functions as voterlab
# import Utilities.DataManagerPytorch as DMP
# import Utilities.LoadVoterData as LoadVoterData
import tmpUtils
import DataManagerPytorch

''' Each model and dataset has its own set of training hyperparameters. 

Args:
    useGrayscale: Set to True to train a model on grayscale dataset (one channel), else False for RGB
'''


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

def gradients_heatmap(data, img_size, title, cmap="viridis"):
    """
    Create and save a heatmap from the data into the `figs/` directory one level above the current working directory.

    Args:
        data (torch.Tensor): The data to visualize, reshaped to `img_size`.
        img_size (tuple): The shape of the data to reshape into (height, width).
        title (str): Title for the heatmap.
        cmap (str): Color map to use for the heatmap.
    """
    matrix = data.detach().numpy().mean(axis=0).reshape(img_size[1], img_size[2])
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, cmap=cmap, cbar=True)
    plt.title(title)
    # plt.xlabel("Width")
    # plt.ylabel("Height")

    # Define the save path for the heatmap
    top_level_dir = os.path.dirname(os.getcwd())  # Go one level up
    figs_dir = os.path.join(top_level_dir, "figs")  # Path to `figs` directory
    os.makedirs(figs_dir, exist_ok=True)  # Ensure `figs` directory exists

    # Save the figure in the `figs` directory
    save_path = os.path.join(figs_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(save_path, dpi=900, bbox_inches='tight')  # Save with high resolution
    plt.close()  # Close the figure to free memory

    print(f"Heatmap saved to: {save_path}")

def TrainBubbleSVM(useGrayscale):
    # Hyperparameters
    imgSize = ((1, 40, 50) if useGrayscale else (3, 40, 50))
    batchSize = 1
    print("------------------------------------")
    # Get dataloaders
    trainLoader, valLoader = tmpUtils.ReturnVoterLabDataLoaders(imgSize = imgSize, loaderCreated = True, batchSize = batchSize, loaderType = 'BalBubbles')
    xtrain, ytrain =  tmpUtils.DataLoaderToTensor(trainLoader)
    xtest, ytest = tmpUtils.DataLoaderToTensor(valLoader)
    xtrain = torch.flatten(xtrain, start_dim = 1)
    xtest = torch.flatten(xtest, start_dim = 1)
    # Normalize
    xtrain /= 255
    xtest /= 255
    # Initialize model and train
    model = pseudoSVM(xtrain.size()[1], 1)
    model.TrainModel(xtrain, ytrain, xtest, ytest)
    # Save trained SVM
    saveTag = 'SVM-B'
    saveDir = (saveDirGrayscale if useGrayscale else saveDirRGB)
    torch.save(model.state_dict(), os.path.join(saveDir, saveTag + '.pth'))


def TrainCombinedSVM(useGrayscale):
    # Hyperparameters
    imgSize = ((1, 40, 50) if useGrayscale else (3, 40, 50))
    batchSize = 1
    print("------------------------------------")
    # Get dataloaders
    trainLoader, valLoader = tmpUtils.ReturnVoterLabDataLoaders(imgSize = imgSize, loaderCreated = True, batchSize = batchSize, loaderType = 'BalCombined')
    xtrain, ytrain =  tmpUtils.DataLoaderToTensor(trainLoader)
    xtest, ytest = tmpUtils.DataLoaderToTensor(valLoader)
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

    # Create heatmaps
    gradients_heatmap(
        data=gradients, 
        img_size=(1, 40, 50), 
        title="Gradient Heatmap", 
        # save_path="./heatmaps/gradient_heatmap.png", 
        cmap="viridis"
    )
    # Save trained SVM
    saveTag = 'SVM-C'
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

# NOTE: Place Train + Bubble/Combined + Model Name function here!

# to train Bubble SVM, uncomment this line:
# TrainBubbleSVM(useGrayscale=True)

# to train Combined SVM, uncomment this line:
TrainCombinedSVM(useGrayscale=True)
