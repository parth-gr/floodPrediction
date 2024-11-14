import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle


def retrieve_and_clean_data():
    filename = "train.csv"
    data_df = pd.read_csv(filename)
    # data_df["WillFloodingOccur"] = np.where(data_df["FloodProbability"] < 0.5, 0, 1)

    X = data_df.iloc[:, 1:21].to_numpy()
    y = data_df["FloodProbability"].values

    return train_test_split(X, y, random_state=1)


def predict_flood_prob(request):
    # Load model
    with open('model.pkl', 'rb') as saved_model_file:
        mdl = pickle.load(saved_model_file)

    return mdl.predict(np.reshape(request, (1, -1)))


def print_metrics(X, y, mdl):
    y_pred = mdl.predict(X)

    MAE = metrics.mean_absolute_error(y, y_pred)
    print("Mean absolute error = {}".format(MAE))

    MSE = metrics.mean_squared_error(y, y_pred)
    print("Mean squared error = {}".format(MSE))

    RMSE = metrics.root_mean_squared_error(y, y_pred)
    print("Root mean squared error = {}".format(RMSE))

    R2 = metrics.r2_score(y, y_pred)
    print("R\u00b2 score = {}".format(R2))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = retrieve_and_clean_data()

    # Create model and save
    model = MLPRegressor(random_state=1, hidden_layer_sizes=(25,)).fit(X_train, y_train)
    print("Model trained successful!")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model saved to disk!")

    print_metrics(X_test, y_test, model)
