import wandb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# import seaborn as sns

import os

def retrieve_and_clean_data():
    data_df = pd.read_csv("data/train.csv")
    print(data_df.head())
    data_df.drop('id',axis=1,inplace=True)
    X = data_df.drop(columns= ['FloodProbability']) #features
    y = data_df['FloodProbability'] #target
    return X, y

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
    wandb.init(project="flood-detection-lr")
    
    X, y = retrieve_and_clean_data()
    wandb.log({"data_summary": wandb.Table(data=X[:10], columns=[f"Feature_{i}" for i in range(X.shape[1])])})
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.85,random_state=0)
    wandb.log({"training_samples": len(X_train), "testing_samples": len(X_test)})

    lr = LinearRegression()
    
    lr.fit(X_train,y_train)
    y_pred_lr = lr.predict(X_test)
    
    mape = mean_absolute_percentage_error(y_test,y_pred_lr)
    print("Error of Linear Regression Model = %.2f"%(mape*100),'%')
    print("Accuracy of Linear Regression Model = %.2f"%((1 - mape)*100),'%')
    r2 = r2_score(y_test,y_pred_lr)
    print("R2 score of Linear Regression = %.2f"%(r2))
    wandb.log({"train_mse": mean_squared_error(y_test, y_pred_lr), "train_r2": r2})

    metrics_dict = calculate_metrics(X_train, X_test, y_train, y_test, lr)
    wandb.log({"test_mse": metrics_dict["Test MSE"], "test_r2": metrics_dict["Test R2"]})
    wandb.log({"mae": metrics_dict["MAE"]})
    wandb.log({"rmse": metrics_dict["RMSE"]})
    wandb.log({"explained_variance_score": metrics_dict["EVS"]})
    wandb.log({"Max error": metrics_dict["ME"]})

    # Log residuals
    residuals_table = wandb.Table(data=list(zip(metrics_dict["y_pred_test"], metrics_dict["residuals"])), columns=["Predicted Values", "Residuals"])
    wandb.log({
        "residuals_scatter": wandb.plot.scatter(
            residuals_table, "Predicted Values", "Residuals", title="Predicted Values vs Residuals"
        )
    })
