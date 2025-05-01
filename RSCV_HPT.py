# Usual suspects
import pandas as pd
import numpy as np

# Random Forest and hyperparameter tuning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_curve, auc

# For plotting the ROC curve
import matplotlib.pyplot as plt

# For saving your trained model
import joblib

# For handling command-line arguments
import argparse

# TensorFlow just for GPU setup — not actually using it for training here
import tensorflow as tf

# Choose which GPU to use if you have more than one
gpu_num = 0

# Check if there are any GPUs and make sure TensorFlow doesn't hog all the memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[gpu_num], 'GPU')
    except RuntimeError as e:
        print(e)

# Hide all those annoying sklearn warnings
import warnings
warnings.filterwarnings("ignore")

# Main function — takes in a CSV file and runs the whole training pipeline
def main(data_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Separate features (X) and labels (y)
    X = df.drop(columns=['labels'])
    y = df['labels']

    # Split the data into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define hyperparameter options for tuning — pretty comprehensive
    param_dist = {
        'n_estimators': [100, 200, 300, 500, 700, 1000, 1500],
        'max_depth': [None, 10, 20, 30, 40, 50, 60],
        'min_samples_split': [2, 5, 10, 15, 20],
        'min_samples_leaf': [1, 2, 4, 6, 8],
        'max_features': ['sqrt', 'log2', None, 'auto'],
        'bootstrap': [True, False]
    }

    # Set up the Random Forest classifier
    clf = RandomForestClassifier(random_state=42)

    # Use RandomizedSearchCV to find the best combo of hyperparameters
    random_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_dist,
        n_iter=20,  # How many combinations to try
        cv=5,       # 5-fold cross-validation
        verbose=2,  # Print progress
        scoring='accuracy',
        random_state=42,
        n_jobs=-1   # Use all available CPU cores
    )

    # Actually run the search
    random_search.fit(X_train, y_train)

    # Print out the best combo it found
    print("Best hyperparameters:", random_search.best_params_)

    # Use the best model it found
    best_model = random_search.best_estimator_

    # Predict on the test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Tuned Model Test Accuracy: {accuracy:.2f}")

    # Get predicted probabilities for ROC curve
    y_pred_prob_rf = best_model.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_pred_prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    # Plot and save the ROC curve
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

    # Save the trained model to a file so you can reuse it later
    joblib.dump(best_model, "best_rf_model.pkl")

# Entry point — allows this script to be run from the command line with --data argument
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with features and label column")
    args = parser.parse_args()
    main(args.data)
