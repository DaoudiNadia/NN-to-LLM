"""
Extract NN knowledge for the adult dataset.

Prepares and loads preprocessed data, trains or loads the
neural network, then runs feature importance analysis, 
feature interaction estimation, and cluster-based pattern 
extraction on the training data.
"""
import os
import torch
import numpy as np
from utils import (prepare_data_adult, train_nn, NN, print_clusters,
                   print_feature_importance, print_feature_interaction,
                   aggregate_interactions,
                   gradient_covariance_interactions)

NPZ_PATH = "processed_data/adult_data_processed.npz"
MODEL_PATH = "trained_nns/adult_nn.pth"

def main():
    """
    Loads or prepares the adult dataset, trains or loads the neural
    network, and prints feature importance, feature interactions, and
    cluster-based pattern summaries.
    """
    if not os.path.exists(NPZ_PATH):
        print("Preprocessed data not found. Preparing data...")
        prepare_data_adult()
    else:
        print(f"Data file '{NPZ_PATH}' already exists. Loading directly.")

    data = np.load(NPZ_PATH, allow_pickle=True)

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_num_orig_train = data["x_num_orig_train"]
    x_cat_orig_train = data["x_cat_orig_train"]
    x_train_orig_all = data["x_train_orig_all"]
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)

    meta = data.get('metadata', {})
    meta = meta.item()
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    ohe_categories = meta["ohe_categories"]
    feature_names = meta["feature_names"]
    feature_names_orig = meta["feature_names_orig"]
    print("Data loaded successfully.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}, skipping training.")
        input_dim = x_train.shape[1]
        model = NN(input_dim).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        model = train_nn(x_train, y_train, model_path=MODEL_PATH,
                         batch_size=512, epochs=200, lr=0.001)

    print_feature_importance(model, x_train, x_num_orig_train,
                                    x_cat_orig_train, num_cols, cat_cols)
    print("\n")
    i_matrix = gradient_covariance_interactions(model, x_train_tensor)
    i_df_agg = aggregate_interactions(i_matrix, num_cols, cat_cols,
                                      feature_names, ohe_categories)
    print_feature_interaction(i_df_agg, top_n=5)
    print("\n")
    print_clusters(model, x_train, x_train_orig_all, x_num_orig_train,
                   x_cat_orig_train, feature_names_orig, meta, y_train)


if __name__ == "__main__":
    main()
