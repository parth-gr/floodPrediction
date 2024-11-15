import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import pickle


def retrieve_and_clean_data():
    filename = "data/train.csv"
    data_df = pd.read_csv(filename)
    # data_df["WillFloodingOccur"] = np.where(data_df["FloodProbability"] < 0.5, 0, 1)

    X = data_df.iloc[:, 1:21].to_numpy()
    y = data_df["FloodProbability"].values

    return X, y


def predict_flood_prob(request):
    # Load model
    with open('model.pkl', 'rb') as saved_model_file:
        mdl = pickle.load(saved_model_file)

    return mdl.predict(np.reshape(request, (1, -1)))


if __name__ == '__main__':
    X, y = retrieve_and_clean_data()

    # Create model and save
    model = MLPRegressor(random_state=1, hidden_layer_sizes=(25,)).fit(X, y)
    print("Model trained successful!")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Model saved to disk!")

