# Sampling Assignment

## Overview
This project focuses on implementing data sampling techniques and evaluating the performance of various machine learning models. The aim is to address class imbalance and identify the most effective sampling strategies for different models.

## Steps to Run the Code

### Prerequisites
Ensure your system meets the following requirements:
- Python 3.8 or newer
- Required libraries (see the next section)

### Required Libraries
Install the necessary libraries using the command below:
```bash
pip install pandas scikit-learn imbalanced-learn
```

### Dataset
The dataset can be accessed and downloaded from the following link:  
[Creditcard_data.csv](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv?raw=true)

### Running the Script
Follow these steps to execute the script:
1. Clone or download this repository.
2. Ensure the dataset is available (the script fetches it automatically if needed).
3. Run the script with the following command:
   ```bash
   python sampling_assignment.py
   ```
4. Analyze the output to identify the best sampling technique for each model.

## Key Components

### Dataset Balancing
The `RandomOverSampler` from the `imblearn` library is used to balance the dataset, addressing class imbalance issues.


### Machine Learning Models
The following models are utilized:
1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Classifier (SVC)
4. Gaussian Naive Bayes
5. K-Nearest Neighbors (KNN)

### Evaluation Metrics
The performance of each model is assessed using accuracy scores for the different sampling techniques. The results are then analyzed to identify the optimal approach for each model.

## Output
The script outputs the best sampling technique for each model based on accuracy scores. Example:

**Best techniques for each model:**
- Logistic Regression: Sampling Technique 1  
- Random Forest Classifier: Sampling Technique 3  
- Support Vector Classifier (SVC): Sampling Technique 4  
- Gaussian Naive Bayes: Sampling Technique 2  
- K-Nearest Neighbors (KNN): Sampling Technique 5

