import os
from PIL import Image, ImageDraw
import re
import torch


''' Given a directory of extracted bubbles, creates another directory of said bubbles pasted on pages 

1000 bubbles will produce 11 pages, first 10 pages have 96 bubbles each

Args:
    fileName: Specific directory for extracted bubbles, for iterating through model and eps value
    bubble_directory: Base directory for extracted bubbles
    misclassificationLabels: Boolean for extrated bubble title, set to True if contains __(Correct|Misclassified)
    saveDir: Directory to save pages
'''


# Constants for sheet layout
dpi = 200
PAGE_WIDTH = int(8.5 * dpi)  # 8.5 inches converted to pixels at 200 dpi
PAGE_HEIGHT = int(11 * dpi)  # 11 inches converted to pixels at 200 dpi
MARGIN = int(1 * dpi)  # 1 inch margin converted to pixels at 200 dpi
SPACING = int(0.5 * dpi)  # 0.5 inch spacing converted to pixels at 200 dpi
BUBBLE_WIDTH = 50  # Bubble width in pixels
BUBBLE_HEIGHT = 40  # Bubble height in pixels

# Function to extract batch_index and example_index from the filename
def get_batch_and_example_indices(filename):
    if misclassificationLabels:
        pattern = r"(\d+)th Batch (\d+)th Example__(Correct|Misclassified)_(Vote|Non-Vote).png" 
    else: 
        pattern = r"__(\d+)th Batch (\d+)th Example__(Vote|Non-Vote).png" 
    match = re.match(pattern, filename)

    if match:
        batch_index = int(match.group(1))
        example_index = int(match.group(2))
        return batch_index, example_index
    else:
        raise ValueError(f"Invalid filename format: {filename}")

# Function to extract batch_index, example_index, and label from the filename
def get_batch_example_and_label(filename):
    if misclassificationLabels:
        pattern = r"(\d+)th Batch (\d+)th Example__(Correct|Misclassified)_(Vote|Non-Vote).png" 
    else: 
        pattern = r"__(\d+)th Batch (\d+)th Example__(Vote|Non-Vote).png" 
    match = re.match(pattern, filename)

    if match:
        batch_index = int(match.group(1))
        example_index = int(match.group(2))
        if misclassificationLabels: 
            classification = match.group(3)
            vote_type = match.group(4)
            return batch_index, example_index, classification, vote_type
        else: 
            vote_type = match.group(3)
            return batch_index, example_index, "None", vote_type
    else:
        # Return default values in case the label is not present in the filename
        return -1, -1, "Unknown"


def main(fileName, bubble_directory, misclassificationLabels, saveDir):
    # Get the list of PNG files in the bubble directory, sort them based on batch and example indices
    png_files = [file for file in os.listdir(bubble_directory) if file.endswith(".png")]
    png_files.sort(key=get_batch_and_example_indices)

    # Calculate the number of bubbles that can fit on each sheet
    available_width = PAGE_WIDTH - 2 * MARGIN
    available_height = PAGE_HEIGHT - 2 * MARGIN
    num_columns = available_width // (BUBBLE_WIDTH + SPACING)
    num_rows = available_height // (BUBBLE_HEIGHT + SPACING)
    total_bubbles_per_sheet = num_columns * num_rows

    # Initialize counters
    sheet_index = 1
    bubbles_placed = 0

    # Initialize an empty list to store the labels (0 for "Non-Vote", 1 for "Vote")
    labels = []

    while bubbles_placed < len(png_files):
        # Create a new blank sheet
        sheet = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
        draw = ImageDraw.Draw(sheet)

        # Place the bubbles on the sheet
        x = MARGIN
        y = MARGIN

        for index in range(bubbles_placed, min(bubbles_placed + total_bubbles_per_sheet, len(png_files))):
            if (index - bubbles_placed) % num_columns == 0 and index != bubbles_placed:
                # Move to the next row
                x = MARGIN
                y += BUBBLE_HEIGHT + SPACING

            bubble_file = png_files[index]
            bubble_path = os.path.join(bubble_directory, bubble_file)
            bubble = Image.open(bubble_path)

            sheet.paste(bubble, (x, y))

            # Get the label from the filename and add it to the labels list
            _, _, _, label = get_batch_example_and_label(bubble_file)
            label_value = 0 if label == "Vote" else 1
            labels.append(label_value)

            x += BUBBLE_WIDTH + SPACING

        # Save the sheet in the PreImagesSimpleCNN_Val directory
        # (("0" + str(sheet_index)) if (sheet_index < 10) else str(sheet_index))
        trailing_sheet_index = str(sheet_index).zfill(3)
        output_path = str(saveDir) + "/output_sheet_" + trailing_sheet_index + ".png" 
        #output_path = f"{saveDir}/output_sheet_{sheet_index}.png"
        sheet.save(output_path)

        # Update the counters
        sheet_index += 1
        bubbles_placed += total_bubbles_per_sheet

    print(f"{sheet_index - 1} sheets have been generated in the " + fileName + " directory.")
    #print("Labels array:", labels)
    print("Num labels:" + str(len(labels)))
    # Save labels in .torch file
    torch.save(labels, os.path.join(saveDir, fileName + '.torch'))


if __name__ == '__main__':
    
    #fileNames = ['SVMValidation_Images', "SimpleCNNValidation_Images", "ResNet20Validation_Images", "DenseNetValidation_Images"]
    #fileNames = ['Validation_BalCombined_SVM_Images']
    #fileNames = ['Bubbles', 'Non_Bubbles']
    modelNames = ['TWINS'] # ['SVM', 'SimpleCNN', 'ResNet20', 'DenseNet']
    '''
    for modelName in modelNames:
        fileName = modelName + "_Validation"
        #bubble_directory = "/home/caleb/VoterWork/ImageProcessing/Variable_Epsilon_Denoiser_Embedded_PrePrint_Bubbles/" + modelName + "Validation_Images"
        #bubble_directory = '/home/caleb/VoterWork/TWINS/No_Denoiser_Embedded_TWINS_APGD_PrePrint_Bubbles/TWINS_Validation_Results'
        bubble_directory = '/home/caleb/VoterWork/TWINS/All_Attacks_TWINS/' + modelName + "_Validation_Results"
        misclassificationLabels = False

        # Create the Val directory if it doesn't exist
        saveDir = "/home/caleb/VoterWork/TWINS/All_Attacks_TWINS_PrePrint_Pages/" + modelName + "_Validation"
        os.makedirs(saveDir, exist_ok=True)
        #os.makedirs(saveDir + "_NonBubbles", exist_ok=True)

        # Generate pages
        #main(fileName, bubble_directory + "_Bubbles", misclassificationLabels, saveDir + "_Bubbles")
        #main(fileName, bubble_directory + "_NonBubbles", misclassificationLabels, saveDir + "_NonBubbles")
        main(fileName, bubble_directory, misclassificationLabels, saveDir)
    '''
    
    '''
    #fileNames = ["0.062_Adversarial_BalCombined_" + modelAttackStr + "_Images" for modelAttackStr in ["SimpleCNN_FGSM", "SimpleCNN_PGD", "SimpleCNN_PGD", "ResNet20_FGSM", "ResNet20_PGD", "ResNet20_PGD", "DenseNet_FGSM", "DenseNet_PGD", "DenseNet_PGD", "SimpleCNN_APGD", "ResNet20_APGD", "DenseNet_APGD"]]
    #attackNames = ['FGSM', 'PGD', 'APGD']
    attackNames = ['APGD']
    #modelNames = ['SVM', 'SimpleCNN', 'ResNet20', 'DenseNet']
    epsilonList = [0.0155, 0.031, 0.062, 0.124, 0.248]
    for modelName in modelNames:
        for attackName in attackNames:
            for epsilon in epsilonList:
                if (modelName == 'SVM' and attackName == 'FGSM') or (modelName != 'SVM'):
                    fileName = modelName + "_" + attackName + "_" + str(epsilon) + "_Results"
                    #bubble_directory = "/home/caleb/VoterWork/Models/No_Denoiser_Embedded_TWINS_APGD_PrePrint_Bubbles/" + modelName + "_APGD_" + str(epsilon) + "_Results"  #ImageProcessing/0.062_Adv_Denoiser_Embedded_PositiveGradient/" + fileName
                    bubble_directory = "/home/caleb/VoterWork/TWINS/Denoiser_Embedded_Variable_Epsilon_APGD_TWINS/" + modelName + "_" + attackName + "_" + str(epsilon) + "_Results"
                    misclassificationLabels = True

                    # Create the Val directory if it doesn't exist
                    saveDir = "/home/caleb/VoterWork/TWINS/Denoiser_Embedded_Variable_Epsilon_APGD_TWINS_PrePrint_Pages/" + modelName + "_" + attackName + "_" + str(epsilon)
                    os.makedirs(saveDir, exist_ok=True)
                    #os.makedirs(saveDir + "_No_Bubbles", exist_ok=True)

                    # Generate pages
                    main(fileName, bubble_directory, misclassificationLabels, saveDir)
                    #main(fileName, bubble_directory + "_No_Bubbles_Results", misclassificationLabels, saveDir + "_No_Bubbles")
    '''

    '''
    fileName = 'Denoiser_Training_Examples'
    bubble_directory = '/home/caleb/VoterWork/Main/' + fileName
    misclassificationLabels = False
    saveDir = '/home/caleb/VoterWork/ImageProcessing/Pre_Print_Denoiser_Training_Example_Pages'
    os.makedirs(saveDir, exist_ok = True)
    main(fileName, bubble_directory, misclassificationLabels, saveDir)
    '''

    # Hyperparameters
    epsilon_values = [0.01568, 0.03137, 0.06274, 0.12549, 0.25098, 1.0]
    model_names = ['TWINS-B', 'TWINS-C'] # ['SVM-B', 'SVM-C', 'SimpleCNN-B', 'SimpleCNN-C', 'ResNet-20-B', 'ResNet-20-C']
    base_bubble_dir = os.getcwd() + '//TWINS40x50//PrePrint_Bubbles//'
    base_save_dir = os.getcwd() + '//TWINS40x50//PrePrint_Bubble_Sheets//'

    for epsilon in epsilon_values:
        for model in model_names:
            # Retrieve bubble directory
            fileName = model 
            bubble_directory = base_bubble_dir + model + "//" + str(epsilon) 
            misclassificationLabels = False

            # Create the Val directory if it doesn't exist
            saveDir = base_save_dir + model + "//" + str(epsilon) 
            os.makedirs(saveDir, exist_ok=True)

            main(fileName, bubble_directory, misclassificationLabels, saveDir)