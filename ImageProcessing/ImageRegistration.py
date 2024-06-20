import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import PIL


''' Given a directory of scanned pages, register these pages

Note: Use align_and_save_image if scanned images are .png format, else use align_tiff_and_save_image

Args:
    input_image_path: Path for scanned page
    reference_image_path: Path for page before printing, in save_dir from ExtraSpacePNG.py
    output_dir: Directory for saving registered pages
'''


def align_images(reference_image, image):
    # Find features and match keypoints
    sift = cv2.SIFT_create()
    keypoints_reference, descriptors_reference = sift.detectAndCompute(reference_image, None)
    keypoints_image, descriptors_image = sift.detectAndCompute(image, None)

    # Match keypoints using Brute-Force matcher
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_reference, descriptors_image, k=2)

    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("Insufficient matches found for alignment.")

    # Extract corresponding keypoints
    points_reference = np.float32([keypoints_reference[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_image = np.float32([keypoints_image[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Estimate perspective transformation
    transformation_matrix, _ = cv2.findHomography(points_image, points_reference, cv2.RANSAC, 5.0)

    # Warp the image to align with the reference image
    aligned_image = cv2.warpPerspective(image, transformation_matrix,
                                        (reference_image.shape[1], reference_image.shape[0]))

    return aligned_image


def align_png_images(output_dir, reference_image_path):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the reference image
    reference_image = cv2.imread(reference_image_path)

    # Register the reference image
    registered_image = reference_image.copy()

    # Save the registered reference image
    registered_image_path = os.path.join(output_dir, "registered_reference.png")
    cv2.imwrite(registered_image_path, registered_image)

    print("Registered reference image created.")

    png_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".png") and f != "reference.png"])
    pixel_diffs = []

    for i, png_file in enumerate(png_files, start=1):
        page_image_path = os.path.join(output_dir, png_file)
        page_image = cv2.imread(page_image_path)

        # Align the images
        aligned_image = align_images(reference_image, page_image)

        # Calculate the difference in pixel values
        pixel_diff = np.abs(aligned_image.astype(np.int16) - reference_image.astype(np.int16))
        pixel_diffs.append(np.mean(pixel_diff))

        # Save the aligned image
        aligned_image_path = os.path.join(output_dir, f"aligned_page_{i}.png")
        cv2.imwrite(aligned_image_path, aligned_image)

        print(f"Aligned image created for page {i}.")

        if i == 1:
            # Calculate the difference image
            diff_image = np.abs(page_image.astype(np.int16) - aligned_image.astype(np.int16))

            # Normalize the difference image to the range [0, 255]
            diff_image_normalized = cv2.normalize(diff_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Create a heatmap representation
            plt.imshow(diff_image_normalized, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("Difference Heatmap for Page 1")
            plt.show()

            # Save the difference image
            diff_image_path = os.path.join(output_dir, "difference_page_1.png")
            cv2.imwrite(diff_image_path, diff_image_normalized)

            print("Difference image created for page 1.")

    print("Image alignment completed successfully.")

    # Create a graph of pixel value differences
    plt.plot(range(1, len(pixel_diffs) + 1), pixel_diffs)
    plt.xlabel("Page Number")
    plt.ylabel("Pixel Value Difference")
    plt.title("Difference in Pixel Values between Aligned Images and Reference Image")
    plt.show()


def align_and_save_image(input_image_path, reference_image_path, output_dir):
    # Load the input image
    input_image = cv2.imread(input_image_path)

    # Load the reference image
    reference_image = cv2.imread(reference_image_path)

    # Align the images
    aligned_image = align_images(reference_image, input_image)

    # Save the aligned image
    aligned_image_path = os.path.join(output_dir, os.path.basename(input_image_path))

    print(f"Aligned image saved: {aligned_image_path}")


def align_tiff_and_save_image(input_image_path, reference_image_path, output_dir):
    # Load the input image
    pil_image = PIL.Image.open(input_image_path)
    input_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    img = input_image
    print(f'dtype: {img.dtype}, shape: {img.shape}, min: {np.min(img)}, max: {np.max(img)}')

    # Load the reference image
    reference_image = cv2.imread(reference_image_path)
    img = reference_image
    print(f'dtype: {img.dtype}, shape: {img.shape}, min: {np.min(img)}, max: {np.max(img)}')

    # Align the images
    aligned_image = align_images(reference_image, input_image)

    # Save the aligned images using reference base name
    aligned_image_path = os.path.join(output_dir, os.path.basename(reference_image_path))
    cv2.imwrite(aligned_image_path, aligned_image)

    print(f"Aligned image saved: {aligned_image_path}")

'''
num_pages = 6

for i in range(1,num_pages+1):
    # Specify the input PNG image path, reference image path, and output directory
    input_image_path = f"/home/caleb/VoterWork/ImageProcessing/PositiveGradient_Results/ResNet20_PGD_No_Bubbles/ResNet20_PGD_{i}.png"
    reference_image_path = f"/home/caleb/VoterWork/ImageProcessing/0.062_Adv_Denoiser_Embedded_PositiveGradient_Pages/0.062_Adversarial_BalCombined_ResNet20_PGD_Images_No_Bubbles/output_sheet_0{i}.png"
    output_dir = "/home/caleb/VoterWork/ImageProcessing/PositiveGradient_Registered_Results/Validation_SVM_Bubbles"

    # Create the output directory if it doesn't exist
    print(input_image_path)
    os.makedirs(output_dir, exist_ok=True)

    # Align and save the input image
    align_and_save_image(input_image_path, reference_image_path, output_dir)
'''

if __name__ == '__main__':

    '''
    numPages = 11
    attackNames = ['FGSM']  #'FGSM', 'PGD', 
    epsilonList = [0.0155, 0.031, 0.062, 0.124, 0.248]
    #convertRange = {0.0155:4, 0.031:8, 0.062:16, 0.124:32, 0.248:64}
    modelNames = ['SVM']
    #modelNames = ['TWINS']
    baseDir = "/workspace/caleb/VoterLab/ImageProcessing/Denoiser_Embedded_PostPrint_Variable_Epsilon_SVM" #Denoiser_Embedded_Varying_Epsilon_APGD"
    inputPaths = []
    outputPathBaseDir = "/workspace/caleb/VoterLab/ImageProcessing/Denoiser_Embedded_PostPrint_Variable_Epsilon_SVM_REGISTERED" #Denoiser_Embedded_Varying_Epsilon_APGD_REGISTERED"
    outputPaths = []
    for modelName in modelNames:
        #print(baseDir + "/" + modelName + "_Bubbles")
        #inputPaths.append(baseDir + "/" + modelName + "_Non_Denoiser_Embedded/" + modelName + "_Validation")
        #outputPaths.append(outputPathBaseDir + "/" + modelName + "_Validation")
        for epsilon in epsilonList:
            if modelName == 'SVM':
                #print(convertRange[epsilon])
                inputPaths.append(baseDir + "/" + modelName + "_FGSM_" + str(epsilon))
                outputPaths.append(outputPathBaseDir + "/" + modelName + "_FGSM_" + str(epsilon))
            else:
                for attackName in attackNames: # modelName + "_" + 
                    inputPaths.append(baseDir + "/" + modelName + "_" + attackName + "_" + str(epsilon))
                    outputPaths.append(outputPathBaseDir + "/" + modelName + "_" + attackName + "_" + str(epsilon))

    referencePaths =  []
    baseDir = "/workspace/caleb/VoterLab/ImageProcessing/Denoiser_Embedded_SVM_Variable_Epsilon_FGSM_PrePrint_Pages" #Denoiser_Embedded_Variable_Epsilon_APGD_PrePrint_Pages"
    for modelName in modelNames:
        #referencePaths.append(baseDir + "/" + modelName + "_Validation")
        for epsilon in epsilonList: 
            if modelName == 'SVM':
                referencePaths.append(baseDir + "/" + modelName + "_FGSM_" + str(epsilon))
            else:
                
                if modelName == 'TWINS': 
                    referencePaths.append("/workspace/caleb/VoterLab/TWINS/All_Attacks_TWINS_PrePrint_Pages"  + "/" + modelName + "_FGSM_" + str(convertRange[epsilon]))
                else:
                
                for attackName in attackNames:
                    referencePaths.append(baseDir + "/" + modelName + "_" + attackName + "_" + str(epsilon))
    
    #print(inputPaths)
    for i in range(len(inputPaths)):
        print("On " + str(i) + "th directory...")
        input_image_path = inputPaths[i]
        reference_image_path = referencePaths[i]
        output_dir = outputPaths[i]
        #referenceMethodUse = referenceMethod[i]

        for j in range(1, numPages + 1): 
            input_image_path_used = input_image_path + "/output_sheet_" + ("0" if j < 10 else "") + str(j) + ".png"
            reference_image_path_used = reference_image_path + "/output_sheet_" + ("0" if j < 10 else "") + str(j) + ".png"
            print(input_image_path_used)
            print(reference_image_path_used)

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Align and save the input image
            align_and_save_image(input_image_path_used, reference_image_path_used, output_dir)
    '''

    '''
    numPages = 105
    baseDir = os.getcwd() + "/Denoiser_Training_Examples/Post_Print_Denoiser_Training_Pages" 
    outputPathBaseDir = os.getcwd() + "/Denoiser_Training_Examples/Pre_Print_Denoiser_Training_Pages" 
    output_dir = os.getcwd() + '/Denoiser_Training_Examples/Post_Print_Denoiser_Training_Pages_Registered'
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Align and save the input image
    for pageNum in range(1, numPages + 1):
        input_image_path_used = baseDir + '/output_sheet_' + str(pageNum).zfill(2) + '.png'
        reference_image_path_used = outputPathBaseDir + '/output_sheet_' + str(pageNum).zfill(3) + '.png'
        print(input_image_path_used)
        #print(reference_image_path_used)
        align_and_save_image(input_image_path_used, reference_image_path_used, output_dir)
    '''

    # Open the files in read mode
    for model in ['TWINS-B', 'TWINS-C']: # ['SVM-B', 'SVM-C', 'SimpleCNN-B', 'SimpleCNN-C', 'ResNet-20-B', 'ResNet-20-C']:
        print_order = open(os.getcwd() + '//TWINS40x50//' + model + '_print_order', "r")
        print_order = print_order.read()
        print_order = print_order.split("\n")

        input_index = 0
        for next_print in print_order:
            #print_title = next_print[6:]
            eps = next_print[8:15]

            output_dir = os.getcwd() + '//PostPrint_REGISTERED_Scans//' + model + '//' + eps
            print(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            reference_path = os.getcwd() + '/TWINS40x50/PrePrint_Bubble_Sheets/' + model + '/' + next_print[8:]
            print(reference_path)
            input_path = os.getcwd() + '/TWINS40x50/' + model + '_PostPrint/bubbles-' + str(input_index) + '.tiff'
            print(input_path)

            align_tiff_and_save_image(input_path, reference_path, output_dir)
            input_index += 1