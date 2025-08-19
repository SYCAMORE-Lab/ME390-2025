from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances

from sklearn.model_selection import train_test_split


def preprocess_data(
    df: pd.DataFrame,
    label: str,
    train_size: float = 0.6,
    seed: Optional[int] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Dict[int, str],
]:
    """Transforms data into numpy arrays and splits it into a train and test set

    Args:
        df: Data to split
        label: name of the training label
        train_size: proportion of the data used for training
        seed: random seed

    Returns:
        object: Tuple containing the training features, training label,
            test features, test label, names of the features and map from label to label_name
    """

    # Shuffle and sort the data
    df = df.sample(frac=1, random_state=seed).sort_values(by=label)
    df[label] = df[label].astype("category")

    # Calculate test size proportion based on remaining data after train split
    test_size = 1.0 - train_size

    # Extract features and labels
    X = df.drop(columns=label).to_numpy()
    y = df[label].cat.codes.to_numpy()

    # Stratified split to obtain train and test sets
    if test_size == 0:
        X_train, X_test, y_train, y_test = X, None, y, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )

    # Generate label map and feature names
    label_map = dict(enumerate(df[label].cat.categories))
    feature_names = list(df.drop(columns=label).columns)

    return X_train, y_train, X_test, y_test, feature_names, label_map


def plot_explained_variance(explained_variance):
    """
    Plot the explained variance of each principal component.

    Parameters:
    explained_variance (np.ndarray): Explained variance ratios.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, align='center',
            label='individual explained variance')
    plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xscale('log')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(X_pca, y, label_map):
    """
    Scatter plot of the first two principal components, colored by y with a legend using label_map.

    Parameters:
    X_pca (np.ndarray): PCA-transformed feature matrix.
    y (np.ndarray): Target labels used for coloring the points.
    label_map (dict): Dictionary to map numerical labels to their names.
    """
    # Replace numbers in y with label names
    y_labels = np.array([label_map[val] for val in y])

    unique_labels = np.unique(y_labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(8, 5))
    
    for i, unique_label in enumerate(unique_labels):
        plt.scatter(X_pca[y_labels == unique_label, 0], X_pca[y_labels == unique_label, 1], 
                    color=colors[i], label=unique_label, s=50)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Dataset')
    plt.legend(title='Classes')
    plt.show()
    