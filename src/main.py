import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
import wandb
import os
from sklearn.metrics import mean_squared_error, r2_score


def retrieve_and_clean_data():
    data_df = pd.read_csv("data/train.csv")
    # data_df["WillFloodingOccur"] = np.where(data_df["FloodProbability"] < 0.5, 0, 1)

    X = data_df.iloc[:, 1:21].to_numpy()
    y = data_df["FloodProbability"].values

    return X, y


def predict_flood_prob(request):
    # Load model
    with open('model.pkl', 'rb') as saved_model_file:
        mdl = pickle.load(saved_model_file)

    return mdl.predict(np.reshape(request, (1, -1)))


def calculate_metrics(X_train, X_test, y_train, y_test, mdl):
    metrics_dict = {}
    y_pred_train = mdl.predict(X_train)
    y_pred_test = mdl.predict(X_test)
    metrics_dict["y_pred_test"] = y_pred_test

    metrics_dict["MAE"] = metrics.mean_absolute_error(y_test, y_pred_test)
    print("Mean absolute error = {}".format(metrics_dict["MAE"]))

    metrics_dict["Training MSE"] = metrics.mean_squared_error(y_train, y_pred_train)
    print("Training mean squared error = {}".format(metrics_dict["Training MSE"]))

    metrics_dict["Test MSE"] = metrics.mean_squared_error(y_test, y_pred_test)
    print("Test mean squared error = {}".format(metrics_dict["Test MSE"]))

    metrics_dict["RMSE"] = metrics.root_mean_squared_error(y_test, y_pred_test)
    print("Root mean squared error = {}".format(metrics_dict["RMSE"]))

    metrics_dict["Training R2"] = metrics.r2_score(y_train, y_pred_train)
    print("Training R\u00b2 score = {}".format(metrics_dict["Training R2"]))

    metrics_dict["Test R2"] = metrics.r2_score(y_test, y_pred_test)
    print("Test R\u00b2 score = {}".format(metrics_dict["Test R2"]))

    metrics_dict["EVS"] = metrics.explained_variance_score(y_test, y_pred_test)
    print("Explained variance score = {}".format(metrics_dict["EVS"]))

    metrics_dict["ME"] = metrics.max_error(y_test, y_pred_test)
    print("Max error = {}".format(metrics_dict["ME"]))

    metrics_dict["residuals"] = y_test - y_pred_test

    return metrics_dict


if __name__ == '__main__':
    # start a new wandb run to track this script
    wandb.init(project="flood-detection")
      
    X, y = retrieve_and_clean_data()
    wandb.log({"data_summary": wandb.Table(data=X[:10], columns=[f"Feature_{i}" for i in range(X.shape[1])])})

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    print("Data split into training and testing sets.")
    wandb.log({"training_samples": len(X_train), "testing_samples": len(X_test)})

    # Create model and save
    model = MLPRegressor(random_state=1, hidden_layer_sizes=(25,)).fit(X_train, y_train)
    print("Model trained successful!")

    print("Evaluating model...")
    metrics_dict = calculate_metrics(X_train, X_test, y_train, y_test, model)

    # Plot loss
    training_loss_list = model.loss_curve_
    training_loss_table = wandb.Table(data=list(zip(range(len(training_loss_list)), training_loss_list)), columns=["Epoch", "Loss"])
    wandb.log({
        "Loss during training": wandb.plot.line(
            training_loss_table, "Epoch", "Loss", title="Training loss during training"
        )
    })

    wandb.log({"train_mse": metrics_dict["Training MSE"], "train_r2": metrics_dict["Training R2"]})
    wandb.log({"test_mse": metrics_dict["Test MSE"], "test_r2": metrics_dict["Test R2"]})
    wandb.log({"mae": metrics_dict["MAE"]})
    wandb.log({"rmse": metrics_dict["RMSE"]})
    wandb.log({"explained_variance_score": metrics_dict["EVS"]})
    wandb.log({"Max error": metrics_dict["ME"]})


    # Log the scatter plot
    table = wandb.Table(data=list(zip(y_test, metrics_dict["y_pred_test"])), columns=["True Values", "Predicted Values"])
    wandb.log({
        "prediction_scatter": wandb.plot.scatter(
            table, "True Values", "Predicted Values", title="True vs Predicted Values"
        )
    })

    # Log residuals
    residuals_table = wandb.Table(data=list(zip(metrics_dict["y_pred_test"], metrics_dict["residuals"])), columns=["Predicted Values", "Residuals"])
    wandb.log({
        "residuals_scatter": wandb.plot.scatter(
            residuals_table, "Predicted Values", "Residuals", title="Predicted Values vs Residuals"
        )
    })

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to disk!")
    
    wandb.save("model.pkl")
    
    wandb.finish()


