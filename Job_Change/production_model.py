import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def get_model():
    #Retrieve training data
    train_data = pd.read_csv('aug_train.csv')
    formatted_data = format_data(train_data)

    #Split the data into x and y datasets, and test-train split
    X_data = formatted_data.drop(columns=['target'])
    y_data = formatted_data['target']
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    #Run the logistic regression model
    lr_model = LogisticRegression(solver='liblinear', random_state=42).fit(X_train, y_train)
    return lr_model, X_train

def format_data(data):
    #Perform basic data cleaning operations
    data = data.drop(columns=['enrollee_id'])
    data['city_development_index'] = 1000 * data['city_development_index']
    data['experience'].replace({'>20': 21, '<1': 0, np.nan: 11}, inplace=True)
    data['company_size'].replace(
        {'<10': 1, '10/49': 2 , 'Oct-49': 2, '50-99': 3, '100-500': 4, '500-999': 5, '1000-4999': 6, '5000-9999': 7, '10000+': 8,
         np.nan: 4}, inplace=True)
    data['last_new_job'].replace({'never': 1, '1': 2, '2': 3, '3': 4, '4': 5, '>4': 6, np.nan: 3}, inplace=True)
    data = pd.get_dummies(data, columns=['city', 'gender', 'relevent_experience', 'enrolled_university',
                                                     'education_level', 'major_discipline', 'company_type'])
    return data

def format_test_data(training_data, data):
    #Format test data, and include all training data columns in df
    formatted_data = format_data(data)
    column_names = list(training_data.columns)
    complete_data = pd.DataFrame(columns=column_names)
    complete_data = pd.concat([complete_data, formatted_data])
    complete_data = complete_data.fillna(0)
    return complete_data

#Return the predicted output
def evaluate(model, data):
    lr_pred = model.predict_proba(data[:])[:,1:2]
    return lr_pred


def run():
    model = None
    #Print input to user, allowing options for the user to build the model, input a file, and quit the program
    while True:
        command = input('\nTo train or retrain the model, type \"build\".  '
                        'For the model to build, there must be a file present named \"aug_train.csv\".")'
                        '\nWhen the model is built, enter a file.  The file should be a CSV, containing various employee fields.'
                     "To quit, enter \"quit\":\n")
        if command == 'build':
            model, training_data = get_model()
            print('The model has been trained')
            continue
        if command == 'quit':
            break
        if model is None:
            print('\nA model must be built before predictions can be made')
            continue
        try:
            file = open(command, "r")
        except:
            print('The file entered does not exist')
            continue

        data = pd.read_csv(file)
        formatted_data = format_test_data(training_data, data)

        #Add to columns: one with the score, and another with the recommendation.  Using the results, create and save a csv
        results = evaluate(model, formatted_data)
        results = pd.DataFrame(results, columns=['Results'])
        labeled_data = data
        labeled_data['Results'] = results
        labeled_data['Recommendation'] = 'Pursue'
        labeled_data['Recommendation'] = labeled_data['Recommendation'].where(labeled_data['Results'] > .45, other='Do Not Pursue')
        labeled_data.to_csv('Labeled_data.csv')
        print('\nA CVS with the results, called \"Labeled_data.csv\", has been created')
    print('Goodbye :)')

run()
