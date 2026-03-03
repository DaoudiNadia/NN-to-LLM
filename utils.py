"""
Utility functions for data preprocessing, neural network training,
feature importance analysis, and cluster-based pattern extraction.
"""

from collections import Counter
import os
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from scipy.io import arff

from config import (mapping_DSD122U, NUM_FEATURES, CATEG_FEATURES,
                    NUM_TO_FILTER, special_missing, feat_mapping,
                    mapping_DSDANTA)



def process_num_features(x_num_orig_train: np.ndarray | pd.DataFrame,
                         x_num_orig_test: np.ndarray | pd.DataFrame,
                         num_features: list,
                         y_train: np.ndarray | pd.DataFrame,
                         corr_threshold: float):
    """
    Imputes, optionally filters, and scales numerical features.

    Parameters:
        x_num_orig_train (np.ndarray | pd.DataFrame): Raw numerical 
            training data.
        x_num_orig_test (np.ndarray | pd.DataFrame): Raw numerical 
            test data.
        num_features (list): Names of numerical features.
        y_train (np.ndarray | pd.DataFrame): Training target values
            used for correlation filtering.
        corr_threshold (float): Minimum absolute correlation required
            to retain a feature. Set to 0 to skip filtering.

    Returns:
        tuple: A tuple containing:
            - x_num_imputed_train (ndarray): Imputed training raw data.
            - x_num_imputed_test (ndarray): Imputed test raw data.
            - x_num_scaled_train (ndarray): Imputed and scaled
                training data.
            - x_num_scaled_test (ndarray): Imputed and scaled test 
                data.
    """

    num_imputer = SimpleImputer(strategy="mean")
    x_num_imputed_train = num_imputer.fit_transform(x_num_orig_train)
    x_num_imputed_test = num_imputer.transform(x_num_orig_test)

    if corr_threshold > 0:
        x_num_train_df = pd.DataFrame(x_num_imputed_train,
                                      columns=num_features)
        y_train_series = pd.Series(y_train.ravel(), name="target")
        corr = x_num_train_df.corrwith(y_train_series).abs().sort_values(
            ascending=False)
        selected_num_features = corr[corr > corr_threshold].index.tolist()
        x_num_imputed_train = x_num_train_df[selected_num_features].values
        x_num_test_df = pd.DataFrame(x_num_imputed_test, columns=num_features)
        x_num_imputed_test = x_num_test_df[selected_num_features].values
        num_features = selected_num_features

    scaler = StandardScaler()
    x_num_scaled_train = scaler.fit_transform(x_num_imputed_train)
    x_num_scaled_test = scaler.transform(x_num_imputed_test)
    return (x_num_imputed_train, x_num_imputed_test, x_num_scaled_train,
            x_num_scaled_test, num_features)



def process_cat_features(x_cat_orig_train_df: np.ndarray | pd.DataFrame,
                         x_cat_orig_test_df: np.ndarray | pd.DataFrame,
                         categ_features: list):
    """
    Imputes and one-hot encodes categorical features.

    Parameters:
        x_cat_orig_train_df (np.ndarray | pd.DataFrame): Raw
            categorical training data.
        x_cat_orig_test_df (np.ndarray | pd.DataFrame): Raw
            categorical test data.
        categ_features (list): Names of categorical features.

    Returns:
        tuple: A tuple containing:
            - x_cat_imputed_train (ndarray): Imputed training raw data.
            - x_cat_imputed_test (ndarray): Imputed test raw data.
            - x_cat_ohe_train (ndarray): Imputed and encoded training
                data.
            - x_cat_ohe_test (ndarray): Imputed and encoded test data.
            - ohe_feature_names (list): One-hot encoded feature names.
            - ohe_categories (list): Categories found per feature
                during fit.
    """
    cat_imputer = SimpleImputer(strategy="most_frequent")
    x_cat_imputed_train = cat_imputer.fit_transform(x_cat_orig_train_df)
    x_cat_imputed_test = cat_imputer.transform(x_cat_orig_test_df)

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_cat_ohe_train = ohe.fit_transform(x_cat_imputed_train)
    x_cat_ohe_test = ohe.transform(x_cat_imputed_test)

    ohe_feature_names = ohe.get_feature_names_out(categ_features).tolist()
    ohe_categories = [list(cat) for cat in ohe.categories_]

    return (x_cat_imputed_train, x_cat_imputed_test, x_cat_ohe_train,
            x_cat_ohe_test, ohe_feature_names, ohe_categories)



def process_and_save(x_train: np.ndarray | pd.DataFrame,
                     x_test: np.ndarray | pd.DataFrame,
                     y_train: np.ndarray | pd.DataFrame,
                     y_test: np.ndarray | pd.DataFrame,
                     num_features: list,
                     npz_path: str, categ_features: list,
                     corr_threshold: float=0):
    """
    Processes numerical and categorical features and saves the results.

    Calls process_num_features and process_cat_features to impute,
    filter, and encode the data, then saves all arrays and metadata 
    to a compressed .npz file.

    Parameters:
        x_train (np.ndarray | pd.DataFrame): Raw training data.
        x_test (np.ndarray | pd.DataFrame): Raw test data.
        y_train (np.ndarray | pd.DataFrame): Training target.
        y_test (np.ndarray | pd.DataFrame): Test target.
        num_features (list): Names of numerical features.
        npz_path (str): Path where the .npz file will be saved.
        categ_features (list): Names of categorical features.
        corr_threshold (float): Minimum absolute correlation required
            to retain a feature. Defaults to 0 (no filtering).

    Returns:
        None, but saves a compressed .npz file containing the 
        transformed arrays and a metadata dictionary.
    """

    ensure_dir("processed_data")
    npz_path = os.path.join("processed_data", npz_path)
    x_num_orig_train = x_train[num_features].values.astype(float)
    x_num_orig_test = x_test[num_features].values.astype(float)

    (x_num_imputed_train, x_num_imputed_test,
     x_num_scaled_train, x_num_scaled_test,
     num_features) = process_num_features(
         x_num_orig_train, x_num_orig_test, num_features, y_train,
         corr_threshold=corr_threshold)

    if len(categ_features) > 0:
        x_cat_orig_train_df = x_train[categ_features]
        x_cat_orig_test_df = x_test[categ_features]

        (x_cat_imputed_train, x_cat_imputed_test,
        x_cat_ohe_train, x_cat_ohe_test,
        ohe_feature_names, ohe_categories) = process_cat_features(
            x_cat_orig_train_df, x_cat_orig_test_df, categ_features)
    else:
        x_cat_imputed_train = np.empty((x_train.shape[0], 0))
        x_cat_imputed_test = np.empty((x_test.shape[0], 0))
        x_cat_ohe_train = np.empty((x_train.shape[0], 0))
        x_cat_ohe_test = np.empty((x_test.shape[0], 0))
        ohe_feature_names = []
        ohe_categories = []

    x_transformed_train = np.hstack([x_num_scaled_train, x_cat_ohe_train])
    x_transformed_test = np.hstack([x_num_scaled_test, x_cat_ohe_test])
    x_train_orig_all = np.hstack([x_num_imputed_train, x_cat_imputed_train])
    x_test_orig_all = np.hstack([x_num_imputed_test, x_cat_imputed_test])

    meta = {
        "feature_names_orig": num_features + categ_features,
        "feature_names": num_features + ohe_feature_names,
        "num_cols": num_features,
        "cat_cols": categ_features,
        "ohe_categories": ohe_categories,
        "npz_path": npz_path
    }

    np.savez_compressed(
        npz_path,
        x_train=x_transformed_train,
        x_test=x_transformed_test,
        y_train=y_train,
        y_test=y_test,
        x_cat_orig_train=x_cat_imputed_train,
        x_test_orig_all=x_test_orig_all,
        x_num_orig_train=x_num_imputed_train,
        x_train_orig_all=x_train_orig_all,
        metadata=meta
    )

    print(f"Preprocessing data is finished, data saved to {npz_path}")


def prepare_data_churn():
    """
    Prepares and processes the churn dataset.

    Loads the churn CSV file, drops irrelevant columns, splits the data
    into train and test sets, and calls process_and_save to preprocess
    and store the results.

    Returns:
        None, but saves the processed data to 
        'churn_data_processed.npz'.
    """
    csv_path = "data/Churn_Modelling.csv"
    target = "Exited"
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
    num_features_ = ["CreditScore", "Age", "Tenure",
                    "Balance", "NumOfProducts", "EstimatedSalary"]
    binary_features = ["HasCrCard", "IsActiveMember"]
    categ_features = ["Geography", "Gender"]
    feature_cols = num_features_ + binary_features + categ_features

    x = df[feature_cols]
    y = df[target].values.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, stratify=y, random_state=42
    )
    num_features = num_features_ + binary_features

    npz_path = "churn_data_processed.npz"
    process_and_save(x_train, x_test, y_train, y_test, num_features,
                     npz_path, categ_features)


def prepare_data_adult():
    """
    Prepares and processes the Adult dataset.

    Fetches the Adult dataset from OpenML, binarizes the target 
    to indicate income above 50K, splits the data into train and 
    test sets, and calls process_and_save to preprocess and store 
    the results.

    Returns:
        None, but saves the processed data to 
        'adult_data_processed.npz'.
    """
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    x_df = adult.data.copy() # pylint: disable=no-member
    y_ser = (adult.target == ">50K").astype(int).to_frame() # pylint: disable=no-member

    rows = np.arange(len(x_df))
    train_idx, test_idx = train_test_split(
        rows, test_size=0.1, random_state=42, stratify=y_ser
    )

    x_train = x_df.iloc[train_idx].reset_index(drop=True)
    x_test  = x_df.iloc[test_idx].reset_index(drop=True)
    y_train = y_ser.iloc[train_idx].to_numpy(dtype=np.float32)
    y_test  = y_ser.iloc[test_idx].to_numpy(dtype=np.float32)
    num_features = (
        x_df
        .select_dtypes(include=["int64", "float64"])
        .columns
        .tolist()
    )
    categ_features = [c for c in x_df.columns if c not in num_features]

    npz_path = "adult_data_processed.npz"
    process_and_save(x_train, x_test, y_train, y_test, num_features,
                     npz_path, categ_features)



def prepare_data_creditg():
    """
    Prepares and processes the creditg dataset.

    Fetches the creditg dataset from OpenML, binarizes the target 
    to indicate credit worthiness, splits the data into train and 
    test sets, and calls process_and_save to preprocess and store 
    the results.

    Returns:
        None, but saves the processed data to 
        'creditg_data_processed.npz'.
    """
    credit = fetch_openml(name="credit-g", version=1, as_frame=True)
    x_df = credit.data.copy() # pylint: disable=no-member
    y_ser = (credit.target == "good").astype(int).to_frame() # pylint: disable=no-member

    # Split first
    rows = np.arange(len(x_df))
    train_idx, test_idx = train_test_split(
        rows, test_size=0.1, random_state=42, stratify=y_ser
    )

    x_train = x_df.iloc[train_idx].reset_index(drop=True)
    x_test  = x_df.iloc[test_idx].reset_index(drop=True)
    y_train = y_ser.iloc[train_idx].to_numpy(dtype=np.float32)
    y_test  = y_ser.iloc[test_idx].to_numpy(dtype=np.float32)

    num_features = (
        x_df
        .select_dtypes(include=["int64", "float64"])
        .columns
        .tolist()
    )
    categ_features = [c for c in x_df.columns if c not in num_features]

    npz_path = "creditg_data_processed.npz"
    process_and_save(x_train, x_test, y_train, y_test, num_features,
                     npz_path, categ_features)


def prepare_data_higgsmall():
    """
    Prepares and processes the Higgsmall dataset.

    Loads the dataset from a local ARFF file, splits the data into 
    train and test sets, and calls process_and_save to preprocess 
    and store the results. A correlation threshold of 0.01 is applied 
    to filter out weakly correlated numerical features.

    Returns:
        None, but saves the processed data to 
        'higgsmall_data_processed.npz'.
    """
    arff_path = "data/higgs.arff"
    target = "class"

    data, _ = arff.loadarff(arff_path)
    df = pd.DataFrame(data)
    df[target] = df[target].astype(int)

    num_features = [col for col in df.columns if col != target]
    categ_features = []  # No categorical features in HiggsSmall

    x = df[num_features]
    y = df[target].values.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, stratify=y, random_state=42
    )

    npz_path = "higgsmall_data_processed.npz"
    process_and_save(x_train, x_test, y_train, y_test, num_features,
                     npz_path, categ_features, corr_threshold=0.01)



def prepare_data_nhanes():
    """
    Prepares and processes the NHANES dataset.

    Loads the dataset from a local CSV file, replaces dataset-specific
    missing value codes with NaN, drops low-coverage rows and columns, 
    maps coded columns to human-readable categories, and binarizes the 
    target based on a 15 mcg Vitamin D intake threshold. Splits the data 
    into train and test sets and calls process_and_save to preprocess 
    and store the results. A correlation threshold of 0.05 is applied 
    to filter out weakly correlated numerical features.

    Returns:
        None, but saves the processed data to 
        'nhanes_data_processed.npz'.
    """
    csv_path = "data/nhanes.csv"
    target = 'VITD_BIN'

    df = pd.read_csv(csv_path)
    for col, vals in special_missing.items():
        df[col] = df[col].replace(vals, np.nan)

    df['DSD122U'] = df['DSD122U'].map(mapping_DSD122U).astype("object")
    df['DSDANTA'] = df['DSDANTA'].map(mapping_DSDANTA).astype("object")

    df = df.loc[:, df.notna().mean(axis=0) >= 0.1]
    df = df.loc[df.notna().mean(axis=1) >= 0.1]

    df = df[df['DSQIVD'].notna()]
    # 15 mcg threshold for Vitamin D
    df[target] = (df['DSQIVD'] > 15).astype(int)

    df.rename(columns=feat_mapping, inplace=True)
    num_features = [feat_mapping.get(col, col) for col in NUM_FEATURES]
    num_features = [col for col in num_features if col in df.columns]
    categ_features = [feat_mapping.get(col, col) for col in CATEG_FEATURES]
    categ_features = [col for col in categ_features if col in df.columns]
    num_to_filter = [feat_mapping.get(col, col) for col in NUM_TO_FILTER]
    num_to_filter = [col for col in num_to_filter if col in df.columns]

    num_features = num_features + num_to_filter
    feature_cols = num_features + categ_features

    x = df[feature_cols]
    y = df[target].values.reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, stratify=y, random_state=42
    )

    npz_path = "nhanes_data_processed.npz"
    process_and_save(x_train, x_test, y_train, y_test, num_features,
                     npz_path, categ_features, corr_threshold=0.05)


class NN(nn.Module):
    """
    A simple neural network for binary classification.

    Attributes:
        model (nn.Sequential): A sequential model with a hidden layer
            of 512 units, ReLU activation, dropout, and a single 
            output unit.

    Parameters:
        input_size (int): Number of input features.
    """
    def __init__(self, input_size):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        """
        Performs a forward pass through the network.
        """
        return self.model(x)


def train_nn(x_train, y_train, model_path, batch_size=512, epochs=1000,
             lr=0.003):
    """
    Trains the NN and saves the model weights.

    Parameters:
        x_train (np.ndarray): Training input data.
        y_train (np.ndarray): Training target values.
        model_path (str): Path where the model weights will be saved.
        batch_size (int): Number of samples per batch. Defaults to 512.
        epochs (int): Number of training epochs. Defaults to 1000.
        lr (float): Learning rate for the Adam optimizer.
        Defaults to 0.003.

    Returns:
        NN: The trained model.
    """

    ensure_dir("trained_nns")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NN(x_train.shape[1]).to(device)

    train_dataset = TensorDataset(torch.from_numpy(x_train).float(),
                                  torch.from_numpy(y_train).float())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: loss={total_loss / len(train_dataset):.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model

def aggregate_interactions(i_matrix: np.ndarray, num_cols: list,
                           cat_cols: list, feature_names: list,
                           ohe_feature_names: list[list]):
    """
    Aggregates OHE-level interactions into original feature-level
    interactions.

    Normalizes the interaction matrix by its maximum absolute value, maps
    one-hot encoded columns back to their original categorical feature
    names, then averages absolute interaction strengths across all OHE
    column pairs belonging to the same parent feature pair.
    Self-interactions are excluded.

    Parameters:
        i_matrix (np.ndarray): Covariance matrix of shape
            (n_features, n_features).
        num_cols (list): Names of numerical features.
        cat_cols (list): Names of categorical features.
        feature_names (list): Full list of feature names including OHE
            columns, used as index.
        ohe_feature_names (list[list]): OHE category values per 
            categorical feature, aligned with cat_cols.

    Returns:
        pd.DataFrame: A DataFrame with columns 'Feature 1',
            'Feature 2', and 'Interaction strength', sorted by 
            interaction strength descending.
    """
    i_norm = i_matrix / np.max(np.abs(i_matrix))
    i_df = pd.DataFrame(i_norm, index=feature_names, columns=feature_names)

    mapping = {}
    for cat, ohe_list in zip(cat_cols, ohe_feature_names):
        for val in ohe_list:
            mapping[f"{cat}_{val}"] = cat
    for num in num_cols:
        mapping[num] = num

    i_df_abs = i_df.abs().copy()
    agg = {}
    for f1 in i_df_abs.index:
        for f2 in i_df_abs.columns:
            parent1, parent2 = mapping[f1], mapping[f2]
            key = tuple(sorted((parent1, parent2)))
            agg[key] = agg.get(key, []) + [i_df_abs.loc[f1, f2]]

    agg_mean = {k: np.mean(v) for k, v in agg.items() if k[0] != k[1]}
    df = pd.DataFrame(
        [(k[0], k[1], v) for k, v in agg_mean.items()],
        columns=["Feature 1", "Feature 2", "Interaction strength"]
    ).sort_values("Interaction strength", ascending=False)
    return df


def compute_integrated_gradients(model: NN, x_train: np.ndarray,
                                 n_steps: int=50):
    """
    Computes Integrated Gradients for all input features.

    Parameters:
        model (NN): The trained neural network model.
        x_train (np.ndarray): Transformed and scaled training data.
        n_steps (int): Number of steps for the approximation.
            Defaults to 50.

    Returns:
        np.ndarray: Integrated Gradients array of shape
            (n_samples, n_features).
    """
    device = next(model.parameters()).device
    model.eval()
    x_tensor = torch.from_numpy(x_train).float().to(device)
    baseline = torch.zeros_like(x_tensor).to(device)

    # Compute Integrated Gradients
    scaled_diff = x_tensor - baseline
    total_grad = torch.zeros_like(x_tensor)

    for alpha in np.linspace(0, 1, n_steps):
        x_step = baseline + alpha * scaled_diff
        x_step.requires_grad_(True)
        logits = model(x_step)
        logit_sum = logits.sum()
        model.zero_grad()
        logit_sum.backward()
        grads = x_step.grad
        total_grad += grads.detach()

    avg_grad = total_grad / n_steps
    return (scaled_diff * avg_grad).cpu().numpy()


def summarize_num_feature_importance(ig: np.ndarray, x_num_orig: np.ndarray,
                                     num_cols: list):
    """
    Summarizes Integrated Gradients for numerical features by binning.

    Skips nearly constant features. Bins values into quantiles, 
    separating zeros from non-zero values. Only retains bins 
    with an absolute mean IG above 0.05.

    Parameters:
        ig (np.ndarray): Integrated Gradients array of shape
            (n_samples, n_features).
        x_num_orig (np.ndarray): Original unscaled numerical training 
            data.
        num_cols (list): Names of numerical features.

    Returns:
        dict: A dictionary mapping each numerical feature name to its
            importance summary, including effect ranges, mean IG 
            scores, column min/max, and feature type 
            (discrete or continuous).
    """

    summary = {}
    for i, col in enumerate(num_cols):
        vals = x_num_orig[:, i]
        ig_vals = ig[:, i]
        # Skip nearly constant features
        if np.std(vals) < 1e-6:
            continue
        # Bin values into quantiles
        df = pd.DataFrame({"val": vals, "ig": ig_vals})
        max_bins = 10
        unique_vals = np.unique(vals)
        if len(unique_vals) <= max_bins:
            # treat each unique value as a bin
            df["bin"] = df["val"]
        else:
            # separate zeros from non-zero values
            zeros = vals == 0
            nonzeros = vals[~zeros]
            df["bin"] = pd.Series(index=df.index, dtype=object)
            df.loc[zeros, "bin"] = "zero"
            if len(nonzeros) > 0:
                bins_nonzero = pd.qcut(nonzeros, q=max_bins-1,
                                       duplicates="drop")
                df.loc[~zeros, "bin"] = bins_nonzero
        bin_means = df.groupby("bin")["ig"].mean()
        bin_abs_means = df.groupby("bin")["ig"].apply(
            lambda x: np.mean(np.abs(x)))

        # Keep bins with meaningful effect
        meaningful_bins = bin_abs_means[bin_abs_means >= 0.05]
        if meaningful_bins.empty:
            continue

        # Separate positive and negative IG bins
        pos_bins = bin_means.loc[meaningful_bins.index][bin_means > 0]
        neg_bins = bin_means.loc[meaningful_bins.index][bin_means < 0]
        if not pos_bins.empty:
            high_bin_label = pos_bins.idxmax()
            high_vals = df.loc[df["bin"] == high_bin_label, "val"]
            high_range = (float(high_vals.min()), float(high_vals.max()))
        else:
            high_range = None
        if not neg_bins.empty:
            low_bin_label = neg_bins.idxmin()
            low_vals = df.loc[df["bin"] == low_bin_label, "val"]
            low_range = (float(low_vals.min()), float(low_vals.max()))
        else:
            low_range = None
        high_mean = float(pos_bins.max()) if not pos_bins.empty else None
        low_mean  = float(neg_bins.min()) if not neg_bins.empty else None
        n_unique = len(np.unique(vals))
        kind = "discrete" if n_unique <= 15 else "continuous"

        summary[col] = {
            "type": kind,
            "high_effect_range": high_range,
            "low_effect_range": low_range,
            "high_mean_ig": high_mean,
            "low_mean_ig": low_mean,
            "column_min_max": (float(np.min(vals)), float(np.max(vals)))
        }
    return summary


def summarize_cat_feature_importance(ig: np.ndarray,
                                     x_cat_orig_df: np.ndarray,
                                     cat_cols: list, start_idx: int):
    """
    Summarizes Integrated Gradients for categorical features.

    Sums IG scores across one-hot encoded columns for each category
    and computes the mean IG per category value.

    Parameters:
        ig (np.ndarray): Integrated Gradients array of shape
            (n_samples, n_features).
        x_cat_orig_df (np.ndarray): Original categorical training data
            before one-hot encoding.
        cat_cols (list): Names of categorical features.
        start_idx (int): Column index in ig where categorical 
            features begin.

    Returns:
        dict: A dictionary mapping each categorical feature name to its
            per-category mean IG scores.
    """
    summary = {}
    for j, col_name in enumerate(cat_cols):
        # number of categories
        n_cat = len(np.unique(x_cat_orig_df[:, j]))
        categories = np.unique(x_cat_orig_df[:, j])

        cat_indices = range(start_idx, start_idx + n_cat)
        # sum IG across one-hot entries
        cat_ig = ig[:, cat_indices].sum(axis=1)
        cat_vals = x_cat_orig_df[:, j]

        effects = {}
        for cat in categories:
            idx = np.where(cat_vals == cat)[0]
            if len(idx) > 0:
                effects[str(cat)] = float(cat_ig[idx].mean())
        summary[col_name] = {"type": "categorical",
                             "category_effects": effects}
        start_idx += n_cat
    return summary


def print_feature_importance(model: NN, x_train: np.ndarray,
                             x_num_orig: np.ndarray,
                             x_cat_orig_df: np.ndarray,
                             num_cols: list, cat_cols: list,
                             n_steps: int=50, n_truncate: int=1):
    """
    Computes and prints a feature importance summary using Integrated
    Gradients.

    Parameters:
        model (NN): The trained neural network model.
        x_train (np.ndarray): Transformed and scaled training data.
        x_num_orig (np.ndarray): Original unscaled numerical training
            data.
        x_cat_orig_df (np.ndarray): Original categorical training data
            before one-hot encoding.
        num_cols (list): Names of numerical features.
        cat_cols (list): Names of categorical features.
        n_steps (int): Number of steps for Integrated Gradients
            approximation. Defaults to 50.
        n_truncate (int): Number of decimal places to truncate IG
            scores for display. Defaults to 1.

    Returns:
        dict: A dictionary mapping each feature name to its importance
            summary.
    """
    ig = compute_integrated_gradients(model, x_train, n_steps)
    summary = summarize_num_feature_importance(ig, x_num_orig, num_cols)
    start_idx = len(num_cols)
    cat_summary = summarize_cat_feature_importance(
        ig, x_cat_orig_df, cat_cols, start_idx)
    summary.update(cat_summary)
    print_ig_json_summary(summary, n_truncate)
    return summary


def print_ig_json_summary(ig_summary, n: int):
    """
    Prints a human-readable summary of Integrated Gradients 
    feature importance.

    For numerical features, prints the value ranges that increase 
    or decrease predictions along with their mean IG scores. 
    For categorical features, prints the top 5 categories by absolute 
    IG score, grouped into increasing and decreasing effects.

    Parameters:
        ig_summary (dict): A dictionary mapping feature names to their
            importance summary, returned by print_feature_importance.
        n (int): Number of decimal places to truncate IG scores for 
            display. Defaults to 1.

    Returns:
        None, but prints a formatted summary to stdout.
    """
    num_feats = {k:v for k,v in ig_summary.items()
                 if v["type"] in ["continuous","discrete"]}
    cat_feats = {k:v for k,v in ig_summary.items()
                 if v["type"]=="categorical"}
    template1 = "value of {} increases predictions (IG={:.1f})"
    template2 = "{}-{} increases predictions (IG={:.1f})"
    template3 = "value of {} decreases (IG={:.1f})"
    template4 = "{}-{} decreases (IG={:.1f})"
    for feat, info in num_feats.items():
        feat_type = info.get("type", "continuous")
        col_minmax = info.get("column_min_max")
        minmax_str = ""
        if col_minmax:
            min_val, max_val = col_minmax
            minmax_str = f" [{truncate(min_val, n)}-{truncate(max_val, n)}]"

        parts = []
        # High effect
        high_values = info.get("high_effect_range")
        high_ig = info.get("high_mean_ig")
        if high_values is not None and high_ig is not None:
            high1 = truncate(high_values[0], n)
            high2 = truncate(high_values[1], n)
            if feat_type == "continuous":
                if high_values[0] == high_values[1]:
                    parts.append(template1.format(high1, high_ig))
                else:
                    parts.append(template2.format(high1, high2, high_ig))
            else:  # discrete
                parts.append(template1.format(high1, high_ig))

        # Low effect
        low_values = info.get("low_effect_range")
        low_ig = info.get("low_mean_ig")
        if low_values is not None and low_ig is not None:
            low1 = truncate(low_values[0], n)
            low2 = truncate(low_values[1], n)
            if feat_type == "continuous":
                if low_values[0] == low_values[1]:
                    parts.append(template3.format(low1, low_ig))
                else:
                    parts.append(template4.format(low1, low2, low_ig))
            else:  # discrete
                parts.append(template3.format(low1, low_ig))

        print(f"- {feat}{minmax_str}: {', '.join(parts)}")

    for feat, info in cat_feats.items():
        effects = info.get("category_effects", {})
        # sort by absolute IG descending
        sorted_cats = sorted(effects.items(), key=lambda x: abs(x[1]),
                             reverse=True)
        top_cats = sorted_cats[:5]  # show top 5 categories
        increase_parts = []
        decrease_parts = []
        for cat, ig in top_cats:
            if ig > 0:
                increase_parts.append(f"'{cat}' (IG={ig:.1f})")
            elif ig < 0:
                decrease_parts.append(f"'{cat}' (IG={ig:.1f})")
        parts = []
        if decrease_parts:
            parts.append(f"({', '.join(decrease_parts)}) decrease")
        if increase_parts:
            parts.append(f"({', '.join(increase_parts)}) increase")
        print(f"- {feat}: {'; '.join(parts)}")


def truncate(x: float, n: int=1):
    """
    Truncates a float to n decimal places without rounding.

    Parameters:
        x (float): The value to truncate.
        n (int): Number of decimal places to keep. Defaults to 1.

    Returns:
        int | float: An integer if the truncated value is whole, 
            otherwise a float truncated to n decimal places.
    """
    factor = 10 ** n
    x_trunc = int(x * factor) / factor  # truncate without rounding
    if abs(x_trunc - round(x_trunc)) < 1e-8:
        return int(round(x_trunc))
    return x_trunc


def gradient_covariance_interactions(model: NN, x_tensor: torch.Tensor,
                                     batch_size: int=512):
    """
    Estimates feature interactions via gradient covariance.

    Computes gradients of the model output with respect to inputs
    across all samples, then returns the covariance matrix of those 
    gradients as a proxy for feature interactions.

    Parameters:
        model (NN): The trained neural network model.
        x_tensor (torch.Tensor): Input tensor of shape 
            (n_samples, n_features), scaled and one-hot encoded.
        batch_size (int): Number of samples processed per batch.
            Defaults to 512.

    Returns:
        np.ndarray: Covariance matrix of shape 
            (n_features, n_features).
    """
    model.eval()
    device = next(model.parameters()).device
    x_tensor = x_tensor.to(device)

    grads_all = []
    for i in range(0, len(x_tensor), batch_size):
        xb = x_tensor[i:i+batch_size].clone().detach().requires_grad_(True)
        yb = model(xb).squeeze()
        grad = torch.autograd.grad(yb.sum(), xb, retain_graph=False)[0]
        grads_all.append(grad.detach().cpu().numpy())

    grads_np = np.vstack(grads_all)
    i = np.cov(grads_np, rowvar=False)
    return i



def print_feature_interaction(df: pd.DataFrame, top_n: int=5):
    """
    Prints the top N feature interactions sorted by strength.

    Parameters:
        df (pd.DataFrame): Interaction DataFrame with columns
            'Feature 1', 'Feature 2', and 'Interaction strength', 
            as returned by aggregate_interactions.
        top_n (int): Number of top interactions to print.
            Defaults to 5.

    Returns:
        None, but prints the top interactions to stdout.
    """
    # Sort by Interaction strength descending
    df_sorted = df.sort_values(
        "Interaction strength", ascending=False).head(top_n)
    for _, row in df_sorted.iterrows():
        feat1 = row["Feature 1"]
        feat2 = row["Feature 2"]
        strength = row["Interaction strength"]
        print(f"- {feat1} and {feat2} (strength: {round(strength, 1)})")



def label_clusters(summary: list[dict], y_true: np.ndarray,
                   threshold: float=0.8):
    """
    Filters clusters where a single label dominates and annotates them.

    Retains only clusters with at least 100 samples where one label
    accounts for at least the given threshold fraction of samples.
    Adds the dominant label and its fraction to the cluster dictionary.

    Parameters:
        summary (list of dict): Per-cluster summary as returned by
            extract_patterns_for_clustering.
        y_true (np.ndarray): True labels for all training samples.
        threshold (float): Minimum fraction required for a label to be
            considered dominant. Defaults to 0.8.

    Returns:
        list of dict: Filtered cluster summaries with added
            'dominant_label' and 'dominant_fraction' keys.
    """
    filtered = []
    for cluster in summary:
        indices = cluster["cluster_indices"]
        if len(indices) >= 100:
            cluster_labels = y_true[indices].ravel()
            counts = Counter(cluster_labels)
            total = len(cluster_labels)
            most_common_label, count = counts.most_common(1)[0]
            fraction = count / total
            if fraction >= threshold:
                cluster["dominant_label"] = int(most_common_label)
                cluster["dominant_fraction"] = fraction
                filtered.append(cluster)
    return filtered


def print_clusters(model: NN, x_train: np.ndarray,
                   x_train_orig_all: np.ndarray, x_num_orig_train: np.ndarray,
                   x_cat_orig_train: np.ndarray, feature_names_orig: list,
                   meta: dict, y_train: np.ndarray, n_truncate: int=1):
    """
    Finds and prints the representative labeled clusters.

    Iteratively increases the number of clusters until at least one
    dominant cluster is found per label (up to 200 clusters). For each
    label, retains the largest dominant cluster and prints its
    feature summary via generate_labeled_clusters.

    Parameters:
        model (NN): The trained neural network model.
        x_train (np.ndarray): Scaled and one-hot encoded training data.
        x_train_orig_all (np.ndarray): Original unscaled training data
            before one-hot encoding.
        x_num_orig_train (np.ndarray): Original unscaled numerical
            training data.
        x_cat_orig_train (np.ndarray): Original categorical training
            data before one-hot encoding.
        feature_names_orig (list): Feature names corresponding to the
            columns of x_train_orig_all.
        meta (dict): Metadata dictionary containing 'num_cols' and
            'cat_cols' keys.
        y_train (np.ndarray): True training labels.
        n_truncate (int): Number of decimal places to truncate IG
            scores for display. Defaults to 1.

    Returns:
        None, but prints cluster summaries to stdout.
    """
    summary_with_labels = []
    n = 10
    while (not summary_with_labels and n<200):
        summary_named, _  = extract_patterns_for_clustering(
            model, x_train, x_train_orig_all, feature_names_orig, meta,
            n_clusters=n)
        print("\n")
        summary_with_labels = label_clusters(
            summary_named, y_train, threshold=0.8)
        n+=10
    print(f"{n} clusters used!")
    clusters_used = []
    if summary_with_labels:
        labels_samples = {elem: [0, {}] for elem in set(y_train.ravel())}
        for cluster in summary_with_labels:
            if cluster["dominant_label"] in labels_samples:
                val = labels_samples[cluster["dominant_label"]]
                if cluster["num_samples"] > val[0]:
                    val[0] = cluster["num_samples"]
                    labels_samples[cluster["dominant_label"]][1] = cluster
    clusters_used = [i[1] for i in labels_samples.values() if i[1]]

    generate_labeled_clusters(
        clusters_used, x_num_orig_train, x_cat_orig_train, meta,
        n = n_truncate
    )




def summarize_features_in_clusters(cluster_num_rows: np.ndarray,
                                   cluster_cat_rows: np.ndarray,
                                   num_indices: list,
                                   discrete_threshold: int=15):
    """
    Summarizes numerical and categorical features for a cluster.

    For numerical features, uses the mode for discrete features and
    the mean for continuous features. For categorical features, 
    the mode is used.

    Parameters:
        cluster_num_rows (np.ndarray): Numerical feature values in 
            the cluster of shape (n_samples, n_num_features).
        cluster_cat_rows (np.ndarray): Categorical feature values in
            the cluster of shape (n_samples, n_cat_features).
        num_indices (list): Column indices of numerical features.
        discrete_threshold (int): Maximum number of unique values 
            for a numerical feature to be treated as discrete.
            Defaults to 15.

    Returns:
        dict: A dictionary mapping column indices to their summary value,
            mean for continuous and mode for discrete and categorical
            features.
    """
    summary = {}
    # Numeric features
    for i, col_idx in enumerate(num_indices):
        values = cluster_num_rows[:, i]
        unique_vals = np.unique(values)
        if len(unique_vals) <= discrete_threshold:
            # Use mode (most frequent value) for discrete features
            summary[col_idx] = Counter(values).most_common(1)[0][0]
        else:
            # Use mean for continuous features
            summary[col_idx] = float(np.mean(values))
    # Categorical features (mode from original)
    for i in range(cluster_cat_rows.shape[1]):
        values = cluster_cat_rows[:, i]
        summary[i + len(num_indices)] = Counter(values).most_common(1)[0][0]

    return summary




def generate_labeled_clusters(summary_with_labels: list[dict],
                              x_num_arr: np.ndarray, x_cat_arr: np.ndarray,
                              meta: dict,
                              discrete_threshold: int=15, n: int=1):
    """
    Generates human-readable descriptions of labeled clusters.

    For each labeled cluster, summarizes numerical and categorical 
    features using summarize_features_in_clusters, then formats them 
    into a descriptive paragraph.

    Parameters:
        summary_with_labels (list of dict): Filtered cluster summaries
            as returned by label_clusters.
        x_num_arr (np.ndarray): Original unscaled numerical training
            data of shape (n_samples, n_num_features).
        x_cat_arr (np.ndarray): Original categorical training data of
            shape (n_samples, n_cat_features).
        meta (dict): Metadata dictionary containing 'num_cols' and
            'cat_cols' keys.
        discrete_threshold (int): Maximum number of unique values for a
            numerical feature to be treated as discrete.
            Defaults to 15.
        n (int): Number of decimal places to truncate IG scores for 
            display. Defaults to 1.

    Returns:
        list of str: One descriptive paragraph per labeled cluster,
            also printed to stdout.
    """
    num_indices = list(range(x_num_arr.shape[1]))
    cat_indices = list(range(x_cat_arr.shape[1]))
    paragraphs = []
    for cluster in summary_with_labels:
        cluster_indices = np.array(cluster["cluster_indices"])
        # slice numeric and categorical arrays
        cluster_num_rows = x_num_arr[cluster_indices, :]
        cluster_cat_rows = x_cat_arr[cluster_indices, :]
        feats_summary = summarize_features_in_clusters(
            cluster_num_rows, cluster_cat_rows, num_indices,
            discrete_threshold)
        feats_rounded = {k: truncate(v, n)
                         if isinstance(v, float)
                         else v for k, v in feats_summary.items()}

        num_feats = ', '.join(f'{meta["num_cols"][i]}: {feats_rounded[i]}'
                                  for i in range(len(num_indices)))
        cat_feats_list = []
        for i in range(len(cat_indices)):
            col_name = meta["cat_cols"][i]
            feat_value = feats_rounded[i + len(num_indices)]
            feat_str = f"{col_name} = {feat_value}"
            cat_feats_list.append(feat_str)
        cat_feats = ', '.join(cat_feats_list)

        label = cluster['dominant_label']
        paragraph = (
            f"{len(cluster_indices)} samples with label {label} "
            f"({cluster['dominant_fraction']*100:.0f}%) have these"
            f"average or dominant features: ({num_feats}); and ({cat_feats})"
        )
        paragraphs.append(paragraph)

    for p in paragraphs:
        print(p)
        print()

    return paragraphs

def find_last_hidden_layer_index(layers: list, min_neurons: int=10):
    """
    Finds the index of the last hidden layer with more than 
    min_neurons units.

    Iterates layers in reverse. Returns the index of the last 
    activation layer whose preceding linear layer has enough 
    units, or the last linear layer with enough units if no 
    activation is found first.

    Parameters:
        layers (list): List of nn.Module layers extracted from 
            the model.
        min_neurons (int): Minimum number of output units required to
            consider a layer as hidden. Defaults to 10.

    Returns:
        int | None: Index of the last valid hidden layer, or None if no
            such layer is found.
    """
    for i in reversed(range(len(layers))):
        layer = layers[i]
        # if activation, return it
        if isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh,
                              nn.ELU)):
            # check preceding linear layer has enough neurons
            if i > 0:
                prev_layer = layers[i-1]
                out_features = getattr(prev_layer, "out_features", None)
                if out_features and out_features > min_neurons:
                    return i
        # if linear layer with enough neurons, return it
        elif isinstance(layer, nn.Linear):
            out_features = getattr(layer, "out_features", None)
            if out_features and out_features > min_neurons:
                return i
    return None



def get_activations_at_layer(model: nn.Module, x_tensor: torch.Tensor):
    """
    Extracts activations at the last hidden layer of the model.

    Forwards the input through the model up to and including the last
    hidden layer.

    Parameters:
        model (nn.Module): The trained neural network model.
        x_tensor (torch.Tensor): Input tensor of shape
            (n_samples, n_features).

    Raises:
        ValueError: If the model contains no sub-layers or no valid 
            hidden layer is found.

    Returns:
        np.ndarray: Activations of shape (n_samples, n_units), 
            where n_units is the size of the last hidden layer.
    """
    layers = get_layer_list(model)
    if len(layers) == 0:
        raise ValueError("Model contains no sub-layers to inspect.")

    layer_idx = find_last_hidden_layer_index(layers)

    if layer_idx is None or layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError("No valid layer found!")

    if any(True for _ in model.parameters()):
        device = next(model.parameters()).device
    else:
        device = torch.device("cpu")
    x = x_tensor.to(device).float()

    # forward through layers
    model.eval()
    with torch.no_grad():
        for i in range(layer_idx + 1):
            layer = layers[i]
            x = layer(x)

    return x.detach().cpu().numpy()



def get_layer_list(model: nn.Module):
    """
    Returns an ordered list of layers to iterate over.

    Prefers model.model if it is an nn.Sequential, then the model
    itself if it is an nn.Sequential, and falls back to
    model.children() otherwise.

    Parameters:
        model (nn.Module): The neural network model.

    Returns:
        list: Ordered list of nn.Module layers.
    """
    if hasattr(model, "model") and isinstance(model.model, nn.Sequential):
        return list(model.model)
    if isinstance(model, nn.Sequential):
        return list(model)
    # fallback: try children (best-effort)
    return list(model.children())


def extract_patterns_for_clustering(model: NN, x_onehot: np.ndarray,
                                    x_train_orig_all: np.ndarray,
                                    feature_names_orig: list, meta: dict,
                                    n_clusters: int=3, pca_components: int=3,
                                    random_state: int=42):
    """
    Clusters training samples based on NN last hidden layer activations
    or the layer itself if no activation is found.

    Extracts activations from the last hidden layer, scales and reduces
    them via PCA, then applies KMeans clustering. For each cluster,
    aggregates numerical features by mean and categorical features
    by mode.

    Parameters:
        model (NN): The trained neural network model.
        x_onehot (np.ndarray): Scaled and one-hot encoded training data.
        x_train_orig_all (np.ndarray): Original unscaled training data
            before one-hot encoding.
        feature_names_orig (list): Feature names corresponding to the
            columns of x_train_orig_all.
        meta (dict): Metadata dictionary containing 'num_cols' and
            'cat_cols' keys.
        n_clusters (int): Number of clusters for KMeans. Defaults to 3.
        pca_components (int): Number of PCA components. Defaults to 3.
        random_state (int): Random seed for reproducibility.
            Defaults to 42.

    Returns:
        tuple: A tuple containing:
            - summary (list of dict): Per-cluster aggregated feature 
                stats, including cluster id, indices, sample count, 
                and mean features.
            - clusters (np.ndarray): Cluster label per training sample 
                of shape (n_samples,).
    """
    model.eval()
    x_tensor = torch.tensor(x_onehot, dtype=torch.float32)
    layer_activations = get_activations_at_layer(model, x_tensor)

    scaler = StandardScaler()
    layer1_scaled = scaler.fit_transform(layer_activations)

    layer1 = layer1_scaled
    pca = PCA(n_components=pca_components, random_state=random_state)
    layer1 = pca.fit_transform(layer1_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(layer1)
    x_train_orig_df = pd.DataFrame(x_train_orig_all,
                                   columns=feature_names_orig)
    # Aggregate per cluster
    summary = []
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        sample_count = len(cluster_indices)
        cluster_rows = [x_train_orig_df.iloc[i] for i in cluster_indices]

        # Aggregate numeric by mean, categorical by mode
        mean_features = {}
        for col in meta["num_cols"]:
            mean_features[col] = float(np.mean([r[col]
                                                for r in cluster_rows]))
        for col in meta["cat_cols"]:
            values = [r[col] for r in cluster_rows]
            mean_features[col] = Counter(values).most_common(1)[0][0]

        summary.append({
            "cluster_id": int(cluster_id),
            "cluster_indices": cluster_indices,
            "num_samples": int(sample_count),
            "mean_features": mean_features
        })

    return summary, clusters

def ensure_dir(path):
    """
    Creates the directory at the given path if it does not exist.

    Parameters:
        path (str): The directory path to create.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)
