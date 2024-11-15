import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
import pickle
import wandb

def retrieve_and_clean_data():
    filename = "../data/train.csv"
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
    # start a new wandb run to track this script
    wandb.init(project="flood-detection")
      
    X, y = retrieve_and_clean_data()
    wandb.log({"data_size": len(X), "feature_count": X.shape[1]})

    # Create model and save
    model = MLPRegressor(random_state=1, hidden_layer_sizes=(25,)).fit(X, y)
    print("Model trained successful!")
    
    train_score = model.score(X, y)  # R^2 score
    wandb.log({"train_score": train_score})
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model saved to disk!")
    wandb.save("model.pkl")
    
    wandb.finish()
