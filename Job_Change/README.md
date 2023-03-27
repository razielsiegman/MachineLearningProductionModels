# Employee Churn Prediction System
This repository contains code for a productionized ML inference system that predicts whether a data scientist is likely to leave their job. The system takes in various features of the employee, and outputs a prediction of whether they are likely to leave their current job or not.

## Dependencies
```
pandas
numpy
sklearn
```

## Running the Model
To run the model, simply execute the employee_churn_prediction.py file. The script will prompt you to enter various commands:

'build': Train or retrain the model. You must have a file named aug_train.csv present in the same directory.
'quit': Exit the program.
'{filename}': Make a prediction on the data contained in the specified CSV file. The file must contain the same columns as aug_train.csv.
When you input a CSV file for prediction, the script will perform some basic data cleaning operations and format the data to match the training data. It will then use the trained logistic regression model to generate a prediction of whether each employee is likely to leave their job or not. The results will be saved to a new CSV file named Labeled_data.csv.

The output file will contain the original input data, as well as two additional columns:

Results: The model's prediction of the likelihood that the employee will leave their job, as a percent.
Recommendation: A recommendation of whether to pursue the employee or not, based on the model's prediction.
