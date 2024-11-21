from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import csv

def retrieve_data():
    data_df = pd.read_csv("data/train.csv")
    data_df = data_df.drop('id', axis=1)

    X = data_df.iloc[:, :20].to_numpy()
    y = data_df["FloodProbability"].values
    return X, y

def hyperparameter_tuning():
    X_features, Y_output = retrieve_data()

    param_grid = { 'hidden_layer_sizes': [(20, 30, 50)],
          'activation': ['relu','tanh','logistic'],
          'alpha': [0.0001, 0.05],
          'learning_rate': ['constant', 'invscaling', 'adaptive'],
          'solver': ['adam', 'sgd'],
          'max_iter':[50, 100, 200] }
    estimator=MLPRegressor()
    gsc = GridSearchCV(
    estimator,
    param_grid,
    cv=5, scoring='r2', verbose=0, n_jobs=-1)

    grid_result = gsc.fit(X_features, Y_output)
    return grid_result

def tune_parameters_and_write():
    gsc_result = hyperparameter_tuning()
    with open('data/hyperParams.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in gsc_result.best_params_.items():
            writer.writerow([key, value])

def read_hyper_parameters():
    params={}
    with open('data/hyperParams.csv') as csv_file:
        reader = csv.reader(csv_file)
        for rows in reader:
            if len(rows):
                print(type(rows[1]))
                if rows[0] == 'hidden_layer_sizes':
                    params[rows[0]] = eval(rows[1])
                elif rows[0] == 'max_iter':
                    params[rows[0]] = eval(rows[1])
                elif rows[0] == 'alpha':
                    params[rows[0]] = eval(rows[1])
                else:
                    params[rows[0]] = rows[1]
    return params

print("Run this file")