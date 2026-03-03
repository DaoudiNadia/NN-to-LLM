"""
Extract NN knowledge for the nhanes dataset.

Prepares and loads preprocessed data, trains or loads the
neural network, then runs feature importance analysis, 
feature interaction estimation, and cluster-based pattern 
extraction on the training data.
"""
import os
import torch
import numpy as np
from utils import (prepare_data_nhanes, train_nn, NN, print_clusters,
                   print_feature_importance, print_feature_interaction,
                   aggregate_interactions,
                   gradient_covariance_interactions)

NPZ_PATH = "processed_data/nhanes_data_processed.npz"
if not os.path.exists(NPZ_PATH):
    print("Preprocessed data not found. Preparing data...")
    prepare_data_nhanes()
else:
    print(f"Data file '{NPZ_PATH}' already exists. Loading directly.")

data = np.load(NPZ_PATH, allow_pickle=True)

x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']
x_num_orig_train = data['x_num_orig_train']
x_cat_orig_train = data['x_cat_orig_train']
x_num_scaled_train = data.get('x_num_scaled_train')
x_train_orig_all = data.get('x_train_orig_all')
meta = data.get('metadata', {})
meta = meta.item()
feature_names_orig = meta['feature_names_orig']
num_cols = meta['num_cols']
cat_cols = meta['cat_cols']
ohe_categories = meta['ohe_categories']
feature_names = meta['feature_names']
feature_names_orig = meta['feature_names_orig']

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)

print("Data loaded successfully.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "trained_nns/nhanes_nn.pth"

if os.path.exists(MODEL_PATH):
    print(f"Model already exists at {MODEL_PATH}, skipping training.")
    input_dim = x_train.shape[1]
    model = NN(input_dim).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    model = train_nn(x_train, y_train, model_path=MODEL_PATH, batch_size=256,
                     epochs=5000, lr=0.01)

summary = print_feature_importance(model, x_train, x_num_orig_train,
                                   x_cat_orig_train, num_cols, cat_cols)
print("\n")

I_matrix = gradient_covariance_interactions(model, x_train_tensor)
I_df_agg = aggregate_interactions(I_matrix, num_cols, cat_cols, feature_names,
                                  ohe_categories)
print_feature_interaction(I_df_agg, top_n=5)
print("\n")

print_clusters(model, x_train, x_train_orig_all, x_num_orig_train,
               x_cat_orig_train, feature_names_orig, meta, y_train)
