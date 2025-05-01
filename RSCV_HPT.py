
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import joblib
import argparse

# TensorFlow is only needed if you want to use GPUs for training
# Import tensorflow for GPU configuration
import tensorflow as tf

gpu_num = 0

# Ensure TensorFlow GPU support
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
    except RuntimeError as e:
        print(e)

import warnings
warnings.filterwarnings("ignore")


def main(data_path):
    # Load Data
    df = pd.read_csv(data_path)
    X = df.drop(columns=['labels'])
    y = df['labels']

    # Split into train/test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # param_dist = {
    #     'n_estimators': [100, 300, 500, 700],
    #     'max_depth': [10, 20, 30, None],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['sqrt', 'log2', None],
    #     'bootstrap': [True, False]
    # }
    
    param_dist = {
        'n_estimators': [100, 200, 300, 500, 700, 1000, 1500],  # Expand range of n_estimators
        'max_depth': [None, 10, 20, 30, 40, 50, 60],  # Increase the max depth options
        'min_samples_split': [2, 5, 10, 15, 20],  # Add more splits for variety
        'min_samples_leaf': [1, 2, 4, 6, 8],  # Add more leaf options
        'max_features': ['sqrt', 'log2', None, 'auto'],  # Include auto as an option
        'bootstrap': [True, False]
    }


    # Initialize Classifier
    clf = RandomForestClassifier(random_state=42)

    # --- HPT with RandomizedSearchCV ---
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        verbose=2,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    # Evaluation
    print("Best hyperparameters:", random_search.best_params_)
    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Tuned Model Test Accuracy: {accuracy:.2f}")

    # ROC & AUC
    y_pred_prob_rf = best_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig("rf_roc_curve.png")
    plt.close()

    # Save Model
    joblib.dump(best_model, "best_rf_model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with features and label column")
    args = parser.parse_args()
    main(args.data)

