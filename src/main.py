import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import wandb
import os
from sklearn.metrics import mean_squared_error, r2_score
from HyperParameterTuning import read_hyper_parameters


def retrieve_and_clean_data():
    data_df = pd.read_csv("data/train.csv")
    X = data_df.iloc[:, 1:21].to_numpy()
    y = data_df["FloodProbability"].values

    return X, y


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
    # start a new wandb run to track this script
    wandb.init(project="flood-detection")
      
    X, y = retrieve_and_clean_data()
    wandb.log({"data_summary": wandb.Table(data=X[:10], columns=[f"Feature_{i}" for i in range(X.shape[1])])})

    params = read_hyper_parameters()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print("Data split into training and testing sets.")
    wandb.log({"training_samples": len(X_train), "testing_samples": len(X_test)})

    # Create model and save
    model = MLPRegressor( random_state=1, hidden_layer_sizes=params['hidden_layer_sizes'],
                         learning_rate=params['learning_rate'],
                         max_iter=params['max_iter'], solver=params['solver'] ).fit(X_train, y_train)

    # model = MLPRegressor(random_state=1, hidden_layer_sizes=(25,)).fit(X_train, y_train)
    print("Model trained successful!")
    
    train_predictions = model.predict(X_train)
    train_mse = mean_squared_error(y_train, train_predictions)
    train_r2 = r2_score(y_train, train_predictions)
    wandb.log({"train_mse": train_mse, "train_r2": train_r2})
    print(f"Training MSE: {train_mse}, R2: {train_r2}")

    print("Evaluating model...")
    test_predictions = model.predict(X_test)
    test_mse = mean_squared_error(y_test, test_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    wandb.log({"test_mse": test_mse, "test_r2": test_r2})
    print(f"Testing MSE: {test_mse}, R2: {test_r2}")
    
    table = wandb.Table(data=list(zip(y_test, test_predictions)), columns=["True Values", "Predicted Values"])

    # Log the scatter plot
    wandb.log({
        "prediction_scatter": wandb.plot.scatter(
            table, "True Values", "Predicted Values", title="True vs Predicted Values"
        )
    })
    
    train_score = model.score(X, y)  # R^2 score
    wandb.log({"train_score": train_score})
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to disk!")
    
    wandb.save("model.pkl")
    
    wandb.finish()

    print_metrics(X_test, y_test, model)
