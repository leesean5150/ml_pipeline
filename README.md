
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
The overarching pipeline consists of a preprocessing step followed by the classifier. This passes the cleaned and processed data into the machine learning model to generate a model to predict whether the call is a scam. There was an attempt to use the Synthetic Minority Over-sampling Technique because of the overwhelming number of non-scam calls in comparison to scam calls, but it caused accuracy and precision of the model to drop significantly, so it was dropped from the planned pipeline. The code however, remains in main.py and there is an option to toggle it on or off in the .env file.\
\
The preprocessing step is a column transformer which helps to handle potential errors in the dataset. This column transformer contains two different pipelines, the categorical processor and the numerical processor, to handle the categorical column and the numerical column respectively.\
\
The categorical processor contains two steps, SimpleImputer and OneHotencoder. The SimpleImputer handles missing/null values in the column, and applies the chosen strategy to it, while the OneHotEncoder handles unkown categories that were not seen in the training set, and applies the chosen strategy to it.\
\
The numerical processor contains two steps as well, SimpleImputer and StandardScaler. The SimpleImputer handles missing/null values in the column, and applies the chosen strategy to it, while the StandardScaler normalizes the data, so that everything is properly scaled.\
\
Before the data is even passed into either the numerical or categorical processor, it also need to be cleaned and transformed. This step includes:
- Removing non_unique IDS
- Transform timestamp into Year, Month, Day and Hour.
- Change Call Type column data from Whats App to Whatsapp.
After all the data is processed, the data is split into training set and test set randomly, and the model is trained using the training set.\
\
Once trained, the model is fed the test set, to try and predict the target, which in this is case is whether or not the call is a scam.\
\
The final step in the pipeline is evaluating the performance of the model using the following performance metrics: accuracy, precision and confusion matrix. This step is repeated a set number of times based on the .env configuration and the average of each metric is returned. This is printed to the console for the user to review.
## Processing of Data
|ID|Call Duration|Call Frequency|Financial Loss|Flagged By Carrier|Is International|Previous Contact Count|Country Prefix|Call Type|Timestamp|Year|Month|Day|Hour|Device Battery|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|SimpleImputer|
|StandardScaler|StandardScaler|StandardScaler|StandardScaler|StandardScaler|StandardScaler|OneHotEncoder|OneHotEncoder|StandardScaler|OneHotEncoder|OneHotEncoder|OneHotEncoder|OneHotEncoder|OneHotEncoder|OneHotEncoder|
|Check for unique values|-|-|-|-|-|-|-|Convert Whats App to Whatsapp|Convert to date and time|-|-|-|-|-|

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

## Choice of Models
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
## Evaluation of Models
There are three methods of evaluation used for this task:
- Accuracy: accuracy talks about how well the model can predict the nature of the call based on the given parameters. A higher accuracy means that the model is able to better predict whether the call is a scam or not which directly solves the problem statement of identifying scam calls. However, this metric cannot be the only metric used to evaluate the model. Since majority of the data entries are non-scam calls, we want to avoid the model just guessing that every call is not a scam call to get a high accuracy. 
- Precision: precision measures the ratio of true positives / total positives. This is an important consideration since we do not want to be classifying non-scam calls as scam calls since it might put the user or caller through unnecessary inconvenience in a real world setting.
- Confusion Matrix: arguably also one of the more important metrics, more specifically, the measure of false negatives and actual negatives. In the real world setting, the repurcussions of a false negative is very high, since the model is classifying a scam call as a non-scam call, which could result in the user being scammed, which is likely to be irreversible.
To average out fluctuations in each run of the model, the average of a set number of runs is returned each time so that the results are more consistent.
### Decision Tree
- Running the default configurations, these are the initial results, Accuracy: 79.05%, Precision: 71.61%, ratio of incorrectly classified scam calls: 28.36%. 
- After optimising, the optimal configuration is:
\
CATEGORICAL_FEATURES: Flagged by Carrier, Is International, Country Prefix, Call Type, Hour\
NUMERIC_FEATURES: Call Duration, Call Frequency, Financial Loss, Previous Contact Count
- This configurations yields the results Accuracy: 79.38%, Precision 72.73%, ratio of incorrectly classified scam calls: 27.21%.
- For the features call duration, is international and previous contact count, the results are different with what was found in the data exploration since the data suggested dropping them due to the lack of correlation, but during the actual model training, the performance was better when these features were included. This could be due to the fact that these features have interaction effects with other features, which helps to improve the model's performance.
### Gradient Boosting
- Running the default configurations, these are the initial results, Accuracy: 83.64%, Precision: 88.70%, ratio of incorrectly classified scam calls: 11.25%. 
- After optimising, the optimal configuration is:
\
CATEGORICAL_FEATURES=Call Type,Year,Month,Day,Hour,Device Battery\
NUMERIC_FEATURES=Call Frequency,Financial Loss\
GRADIENT_NUMBER_OF_DECISIONS_TREES=100\
GRADIENT_LEARNING_RATE=0.2\
- This configurations yields the results Accuracy: 82.95%, Precision 96.11%, ratio of incorrectly classified scam calls: 3.86%.
- For this algorithm, it is more in line with what was found during data exploration, with the only conflict being device battery, which again could be due to the feature having interaction effects with other features, which helps to improve the model's performance.
- Although it can be seen that the model's overall accuracy dropped after optimisation, there was a large increase in the precision and a big drop in false negatives. This is especially important for this task, because even though its accuracy may not be very high, it has very few errors when it comes to true positives and false negatives. Accuracy only takes a hit because of the increase in error rate for true negatives.
### Random Forest
- Running the default configurations, these are the initial results, Accuracy: 82.75%, Precision: 86.94%, ratio of incorrectly classified scam calls: 13.05%. 
- After optimising, the optimal configuration is:
\
CATEGORICAL_FEATURES=Call Type,Year,Month,Day,Hour,Device Battery\
NUMERIC_FEATURES=Call Duration,Call Frequency,Financial Loss,Previous Contact Count
- This configurations yields the results Accuracy: 82.21%, Precision 92.77%, ratio of incorrectly classified scam calls: 7.22%.
- For the features call duration, device battery and previous contact count, the results are different with what was found in the data exploration since the data suggested dropping them due to the lack of correlation, but during the actual model training, the performance was better when these features were included. This could be due to the fact that these features have interaction effects with other features, which helps to improve the model's performance.
- Although it can be seen that the model's overall accuracy dropped after optimisation, there was a large increase in the precision and a big drop in false negatives. This is especially important for this task, because even though its accuracy may not be very high, it has very few errors when it comes to true positives and false negatives. Accuracy only takes a hit because of the increase in error rate for true negatives.
## Conclusion
Overall, the model that had the better performance for this specific task would be the Gradient Boosting algorithm with the optimised configurations. This is because the model boasts much better performance in terms of detection of true positives and prevention of false negatives, protecting the user from scam calls. However, it must be noted that this decrease the number of true negatives, which could cause more non-scam calls to be wrongly classified as a scam call. I feel that this is a more appropriate approach for this task, because false negatives are more detrimental than false positives when it comes to scam calls.
### Further Considerations
- More data should be used to better train the model, allowing the algorithm to improve in terms of accuracy because that is what is lacking now.
- There are many more hyperparameters for the Gradient Boosting algorithm which can be furthur tuned to improve performance.
- A docker container should be considered when it comes to deploying the pipeline to improve reproducibility of the results.
- This pipeline could also be integrated with a web/mobile application to handle inputting of new data to the database as well as toggling features or hyper parameters.