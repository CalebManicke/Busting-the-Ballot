# Busting-the-Ballot
Corresponding repo for "Busting the Ballot with Commodity Printers: Voting Meets Adversarial Machine Learning"

# Repo Overview 
- **Models** Architecture code for SVM, SimpleCNN, ResNet and Twins transformer presented in paper. Denoising Autoencoder architecture is also present here.
- **Train** Training pipeline with hyperparameters for each model across each dataset.
- **Utilities** Helper functions compiled for modifying dataloaders, evaluating model accuracy, converting dataloaders to images, etc.
- **ImageProcessing** Pipeline for creating pages then extracting bubbles from said pages post-print. Broken down into three parts:
  1. ExtraSpacePNG.py takes a directory of bubbles and creates .png pages for printing.
  2. ImageRegistration.py registers a page post-print and scan and aligns it with the pages pre-print.
  3. ExtractBubblesFromWhitespacePNG.py takes registered pages and extracts bubbles.

# Requirements
.yml with necessary libraries are provided. It is worth noting that most dependent libraries are for the Twins model. 
