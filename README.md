# Busting-the-Ballot
Corresponding repo for "Busting the Ballot with Commodity Printers: Voting Meets Adversarial Machine Learning"

# Repo Overview 
\begin{itemize}
\item \textbf{Models: } Architecture code for SVM, SimpleCNN, ResNet and Twins transformer presented in paper. Denoising Autoencoder architecture is also present here.
\item \textbf{Train: } Training pipeline with hyperparameters for each model across each dataset.
\item \textbf{Utilities: } Helper functions compiled for modifying dataloaders, evaluating model accuracy, converting dataloaders to images, etc.
\item \textbf{ImageProcessing: } Pipeline for creating pages then extracting bubbles from said pages post-print. Broken down into three parts:
\begin{enumerate}
\item ExtraSpacePNG.py takes a directory of bubbles and creates .png pages for printing.
\item ImageRegistration.py registers a page post-print and scan and aligns it with the pages pre-print.
\item ExtractBubblesFromWhitespacePNG.py takes registered pages and extracts bubbles.
\end{itemize} 

# Requirements
.yml with necessary libraries are provided. It is worth noting that most dependent libraries are for the Twins model. 
