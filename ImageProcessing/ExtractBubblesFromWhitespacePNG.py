import os
from PIL import Image
import torch


''' Given a directory of registed pages, extract bubbles

Note: Use align_and_save_image if scanned images are .png format, else use align_tiff_and_save_image

Args:
    output_sheet_path: Path for registered page
    num_pages: Number of pages in output_sheet_path directory
    Labels_array: List of labels corresponding to each extracted bubble... found as a .torch file in save_dir from ExtraSpacePNG.py
    extracted_bubbles_directory: Directory for saving extracted bubbles
    numBubbles: Number of bubbles to extract
'''


def ExtractBubbles(output_sheet_path, num_pages, Labels_array, extracted_bubbles_directory, numBubbles):
    labelIndex = 0
    for k in range(1, num_pages + 1):
        # Open the output sheet image
        #output_sheet_path = f"/workspace/caleb/VoterLab/ImageProcessing/Positive_Gradient_PostPrint_Registered/TWINS_Clean/TWINS_Clean_{k}.png"
        output_sheet_page = output_sheet_path + "/output_sheet_" + ("00" if k < 10 else "0") + str(k) + ".png"
        print(output_sheet_page)
        #i = (k-1) * 24
        output_sheet = Image.open(output_sheet_page)

        # Constants for bubble layout
        dpi = 200
        BATCH_SIZE = 64
        PAGE_WIDTH = int(8.5 * dpi)  # 8.5 inches converted to pixels at 200 dpi
        PAGE_HEIGHT = int(11 * dpi)  # 11 inches converted to pixels at 200 dpi
        MARGIN = int(1 * dpi)  # 1 inch margin converted to pixels at 200 dpi
        BUBBLE_WIDTH = 50  # Bubble width in pixels
        BUBBLE_HEIGHT = 40  # Bubble height in pixels
        SPACING = int(0.5 * dpi)  # 0.5 inch spacing converted to pixels at 200 dpi

        # Calculate the number of bubbles that can fit on the sheet
        num_columns = int((PAGE_WIDTH - 2 * MARGIN) / (BUBBLE_WIDTH + SPACING))
        num_rows = int((PAGE_HEIGHT - 2 * MARGIN) / (BUBBLE_HEIGHT + SPACING))
        total_bubbles = num_columns * num_rows
        print("Total Number of Bubbles: " + str(total_bubbles))

        # Directory to save the extracted bubbles
        #extracted_bubbles_directory = "/workspace/caleb/VoterLab/ImageProcessing/Positive_Gradient_Adv_Bubbles/TWINS_CLEAN"

        # Create the directory if it doesn't exist
        os.makedirs(extracted_bubbles_directory, exist_ok=True)

        # Get the number of existing bubbles in the directory
        existing_bubbles = len(os.listdir(extracted_bubbles_directory))

        # Extract the individual bubbles from the sheet
        x = MARGIN
        y = MARGIN

        for row in range(num_rows):
            for col in range(num_columns):
                bubble_box = (x, y, x + BUBBLE_WIDTH, y + BUBBLE_HEIGHT)
                bubble = output_sheet.crop(bubble_box)

                # Save the extracted bubble
                if labelIndex < len(Labels_array):
                    Label = Labels_array[labelIndex]
                    batch_index = (existing_bubbles + row * num_columns + col) // BATCH_SIZE
                    example_index = (existing_bubbles + row * num_columns + col) % BATCH_SIZE
                    bubble_filename = f"{batch_index}th Batch {example_index}th Example_{Label}.png"
                    #print(bubble_filename)
                    bubble_path = os.path.join(extracted_bubbles_directory, bubble_filename)
                    bubble.save(bubble_path)

                x += BUBBLE_WIDTH + SPACING
                #print(i)
                labelIndex+=1

            # Move to the next row
            x = MARGIN
            y += BUBBLE_HEIGHT + SPACING

        '''
        labelIndex = 0
        for row in range(num_rows):
            for col in range(num_columns):
                bubble_box = (x, y, x + BUBBLE_WIDTH, y + BUBBLE_HEIGHT)
                bubble = output_sheet.crop(bubble_box)

                # Save the extracted bubble
                if (labelIndex < numBubbles):
                    Label = Labels_array[labelIndex]
                    batch_index = (existing_bubbles + row * num_columns + col) // BATCH_SIZE
                    example_index = (existing_bubbles + row * num_columns + col) % BATCH_SIZE
                    bubble_filename = str(batch_index) + "th Batch " + str(example_index) + "th Example_" + str(Label) + ".png"
                    #bubble_filename = f"{batch_index}th Batch {example_index}th Example_{Label}.png"
                    #if (batch_index <= 15) and (example_index <= 39):
                    bubble_path = os.path.join(extracted_bubbles_directory, bubble_filename)
                    bubble.save(bubble_path)

                    #y += BUBBLE_HEIGHT + SPACING
                    x += BUBBLE_WIDTH + SPACING
                    #print(labelIndex)
                    labelIndex+=1

            # Move to the next column
            #y = MARGIN + SPACING
            #x += BUBBLE_WIDTH + SPACING
            
            # Move to the next row
            x = MARGIN
            y += BUBBLE_HEIGHT + SPACING
            '''

        print(f"Individual bubbles have been extracted and saved in the {output_sheet_path} directory.")

if __name__ == '__main__':
    '''
    referencePaths =  []
    labels = []
    savePaths = []
    #modelNames = ['SVM', 'SimpleCNN', 'ResNet20', 'DenseNet']
    modelNames = ['TWINS']  # 'SimpleCNN', 'ResNet20', 'DenseNet'] #, 
    attackNames = ['APGD']
    epsilonList = [0.0155, 0.031, 0.062, 0.124, 0.248]
    baseDir = "/workspace/caleb/VoterLab/TWINS/Denoiser_Embedded_TWINS_Variable_Epsilon_APGD_PostPrint_Pages_REGISTERED" #Denoiser_Embedded_Varying_Epsilon_APGD_REGISTERED"
    outputPath = "/workspace/caleb/VoterLab/TWINS/Denoiser_Embedded_TWINS_Variable_Epsilon_APGD_PostPrint_Bubbles" #Denoiser_Embedded_Varying_Epsilon_APGD_PostPrint_Bubbles"
    labelDir = "/workspace/caleb/VoterLab/TWINS/Denoiser_Embedded_Variable_Epsilon_APGD_TWINS_PrePrint_Pages" #Denoiser_Embedded_Variable_Epsilon_APGD_PrePrint_Pages"
    '''

    '''
    for modelName in modelNames:
        
        fileName = modelName + "_Validation"
        savePaths.append(outputPath + "/" + fileName)
        referencePaths.append(baseDir + "/" + modelName + "_Validation")
        checkpointLocation = labelDir + "/" + modelName +"_Validation/" + modelName + "_Validation.torch"
        labelArray = torch.load(checkpointLocation, map_location = torch.device("cpu"))
        #print(modelName + " Validation Labels: " + str(labelArray))
        labels.append(torch.load(checkpointLocation, map_location = torch.device("cpu")))
        
        for epsilon in epsilonList: 
            if modelName == 'SVM':
                fileName = modelName + "_FGSM_" + str(epsilon)
                savePaths.append(outputPath + "/" + fileName)
                referencePaths.append(baseDir + "/" + modelName + "_FGSM_" + str(epsilon))
                checkpointLocation = labelDir + "/" + modelName +"_FGSM_" + str(epsilon) + "/" + modelName + "_FGSM_" + str(epsilon) + "_Results.torch"
                labels.append(torch.load(checkpointLocation, map_location = torch.device("cpu")))
            else:
                for attackName in attackNames:
                    fileName = modelName + "_" + attackName + "_" + str(epsilon)
                    savePaths.append(outputPath + "/" + fileName)
                    referencePaths.append(baseDir + "/" + fileName)
                    checkpointLocation = labelDir + "/" + modelName + "_" + attackName + "_" + str(epsilon) + "/" + fileName + "_Results.torch"
                    labels.append(torch.load(checkpointLocation, map_location = torch.device("cpu")))
    
    for i in range(len(referencePaths)):
        ExtractBubbles(output_sheet_path = referencePaths[i], num_pages = 11, Labels_array = labels[i], extracted_bubbles_directory = savePaths[i], numBubbles = 1000)
    '''

    '''
    referencePath = os.getcwd() + '//Denoiser_Training_Examples//Post_Print_Denoiser_Training_Pages_Registered'
    checkpointLocation = os.getcwd() + '//Denoiser_Training_Examples//Pre_Print_Denoiser_Training_Pages//Denoiser_Training_Examples.torch'
    labels = torch.load(checkpointLocation, map_location = torch.device("cpu"))
    savePath = os.getcwd() + '//Denoiser_Training_Examples//Post_Print_Denoiser_Training_Bubbles'
    ExtractBubbles(output_sheet_path = referencePath, num_pages = 105, Labels_array = labels, extracted_bubbles_directory = savePath, numBubbles = len(labels))
    '''

    for model in ['TWINS-B', 'TWINS-C']: # ['SVM-B', 'SVM-C', 'SimpleCNN-B', 'SimpleCNN-C', 'ResNet-20-B', 'ResNet-20-C']:
        for eps in ['0.01568', '0.03137', '0.06274', '0.12549']:
            reference_path = os.getcwd() + '//PostPrint_REGISTERED_Scans//' + model + '//' + eps 
            labels = torch.load(os.getcwd() + '//TWINS40x50//PrePrint_Bubble_Sheets//' + model + '//' + eps + '//' + model + '.torch', map_location = torch.device("cpu"))
            save_path = os.getcwd() + '//TWINS40x50//PostPrint_Bubbles//' + model + '//' + eps 
            ExtractBubbles(output_sheet_path = reference_path, num_pages = 11, Labels_array = labels, extracted_bubbles_directory = save_path, numBubbles = 1000)