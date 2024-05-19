
# AIIP 3 Techinical Assessment
Name: Lee Sean \
Email: leesean18082001@gmail.com
## Project Directory
```bash
├── .github
├── src
│   ├── main.py
├── .env
├── .gitignore
├── eda.ipynb
├── README.md
├── requirements.txt
└── run.sh
```
## Running Pipeline
```bash
pip install -r requirements.txt
./run.sh
```
## Modifying Environment Variables
The .env file contains the variables to be configured to change things within the pipeline.

- For CATEGORICAL_FEATURES, it indicates the features that are categorical in nature to be used in the model. The default configurations to choose from are: 
Flagged by Carrier,Is International,Country Prefix,Call Type,Year,Month,Day,Hour,Device Battery

- For NUMERIC_FEATURES, it indicates the features that are numerical in nature to be used in the model. The default configurations to choose from are: 
ID,Call Duration,Call Frequency,Financial Loss,Previous Contact Count 

- For CATEGORICAL_FEATURES and NUMERIC_FEATURES please ensure that there are no spaces between the commas. Multiple features can be chosen for these two variables.

- For CATEGORICAL_IMPUTER_STRATEGY, it indicates the strategy to be used to handle null data entries for columns with data that are categorical in nature. The default configurations to choose from are:
mean / median / most_frequent / 0

- For NUMERIC_IMPUTER_STRATEGY, it indicates the strategy to be used to handle null data entries for columns with data that are numerical in nature. The default configurations to choose from are:
mean / median / most_frequent / 0

- For ONEHOT_HANDLE_UNKNOWN, it indicates the strategy to be used to handle categories that were not seen in the training data. The default configurations to choose from are:
error / ignore

- For CATEGORICAL_IMPUTER_STRATEGY, NUMERIC_IMPUTER_STRATEGY and ONEHOT_HANDLE_UNKNOWN, only one option should be used. 

- For TEST_SIZE, it indicates the ratio between the test size and the total size of the data set. The value should be a number between 0 and 1. The default should be left at 0.2.

- For NUMBER_OF_RUNS, it indicates the number of times the pipeline should be ran. The return result of the pipeline would be the average of the number of runs specified here. The default should be left at 10.

- For ADD_SMOTE, it indicated whether or not the Synthetic Minority Over-sampling Technique is applied to the final pipeline. However due to the heavy drop in accuracy and precision, the default should be left at False. The default configurations to choose from are:
True/False\
Only one option should be chosen

- For ALGORITHM, it indicates the algorithm used for building the model. The default configurations to choose from are:
 RandomForest / GradientBoosting / DecisionTree \
 Only one option should be chosen.
 
- If RandomForest is chosen, you may choose to update:
    - NUMBER_OF_TREES, which indicates the number of decision trees grown to train the model. The default should be left at 100.

- If GradientBoosting is chosen, you may choose to update:
    - GRADIENT_NUMBER_OF_DECISIONS_TREES, which indicates the number of decision trees grown to train the model. The default should be left at 100.

    - GRADIENT_LEARNING_RATE, which indicates the amount of contribution that each new tree has for the models overall prediction. The default should be left at 0.1.
## Pipeline Logic
The overarching pipeline consists of a preprocessing step followed by the classifier. This passes the cleaned and processed data into the machine learning model to generate a model to predict whether the call is a scam. There was an attempt to use the Synthetic Minority Over-sampling Technique because of the overwhelming number of non-scam calls in comparison to scam calls, but it caused accuracy and precision of the model to drop significantly, so it was dropped from the planned pipeline. The code however, remains in main.py and there is an option to toggle it on or off in the .env file. This can be seen in the image below:\
\
The preprocessing step is a column transformer which helps to handle potential errors in the dataset. This column transformer contains two different pipelines, the categorical processor and the numerical processor, to handle the categorical column and the numerical column respectively. This can be seen in the image below:\
\
The categorical processor contains two steps, SimpleImputer and OneHotencoder. The SimpleImputer handles missing/null values in the column, and applies the chosen strategy to it, while the OneHotEncoder handles unkown categories that were not seen in the training set, and applies the chosen strategy to it. This can be seen in the image below:\
\
The numerical processor contains two steps as well, SimpleImputer and StandardScaler. The SimpleImputer handles missing/null values in the column, and applies the chosen strategy to it, while the StandardScaler normalizes the data, so that everything is properly scaled. This can be seen in the image below:\
\
Before the data is even passed into either the numerical or categorical processor, it also need to be cleaned and transformed. This step includes:
- Removing non_unique IDS
- Transform timestamp into Year, Month, Day and Hour.
- Change Call Type column data from Whats App to Whatsapp.
After all the data is processed, the data is split into training set and test set randomly, and the model is trained using the training set.\
\
Once trained, the model is fed the test set, to try and predict the target, which in this is case is whether or not the call is a scam.\
\
The final step in the pipeline is evaluating the performance of the model using the following performance metrics: accuracy, precision and confusion matrix. This is printed to the console for the user to review.
## Overview of EDA 
- There is a repeated Call Type, Whats App and Whatsapp
- Financial Loss contains 1403 null entries
- ID contains 2000 non-unique entries
- Financial Loss and Call Duration contains 232 and 2500 negative values respectively.
- Timestamp needs to be converted into a more interpretable data type, which is Year, Month, Day, Hour.
- ID should not be used as a feature because it is just a unique identifier to identify the call.
- Data is inconclusive on whether or not Financial Loss has a correlation with scam calls.
- Data is conclusive that Call Duration has no correlation with scam calls.
- Data is inconclusive on whether or not Call Frequency has a correlation with scam calls.
- Data is conclusive that Flagged By Carrier has correlation with scam calls.
- Data is conclusive that Is International has no correlation with scam calls.
- Data is conclusive that Previous Contact Count has no correlation with scam calls.
- Data is conclusive that Country Prefix has a correlation with scam calls.
- Data is conclusive that Call type has correlation with scam calls.
- Data is conclusive that Year has no correlation with scam calls.
- Data is conclusive that Month has no correlation with scam calls.
- Data is conclusive that Day has no correlation with scam calls.
- Data is conclusive that Hour has a correlation with scam calls.
- Data is conclusive that Battery Device has no correlation with scam calls.
## Processing of Datasets
|Categorical|Numerical|
|-|-|
|SimpleImputer|SimpleImputer|
|OneHotEncoder|StandardScaler|

## Choice of models
Based on the problem statement, the task at hand is a supervised classification task. This is because we are given a labelled dataset (whether or not the call is a scam) and we are tasked to create a model that can classify each call. Because of this, the choice of models are as shown below:
- Decision Tree
    - Decision Tree is a suitable model for this task becasuse it can return a categorical result, making it suitable for the classification task at hand.
    - It handles non-linear relationship between the features better than other potential algorithms, like logistic regression, allowing it to perform better.
    - Despite being a weak learner, the Decision Tree model is much faster compared to the other algorithms chosen, so it is good for quick exploration and initial model prototyping.
- Gradient Boosting
    - Gradient Boosting is a suitable model for this task because it is an ensemble method making use of decision trees, thus gaining the benefits of using a decision tree.
    - On top of the things mentioned above from the decision tree, it also gives higher weightage to errors, since it makes the trees learn in order and correct the previous trees errors. This leads to a more accurate model.
- Random Forest
    - Random Forest is a suitable model for this task because it is an ensemble method making use of decision trees, thus gaining the benefits of using a decision tree.
    - On top of the things mentioned above from the decision tree, it also introduces a random element to the dataset that is fed to each tree. This leads to less overfitting in the model.
