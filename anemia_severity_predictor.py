import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from itertools import cycle # Needed for potential multi-class ROC plotting (though simplified now)
from mpl_toolkits.mplot3d import Axes3D # Needed for 3D plots

# --- 1. Data Loading ---
def load_data(file_path):
    """Loads the dataset from a CSV file."""
    # Use the updated generated anemia dataset file name
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully from {file_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:\n", df.head())
    print("\nDataset Info:\n")
    df.info()
    print("\nTarget variable distribution:\n", df['Anemia_Severity'].value_counts())
    return df

# --- 2. Data Preprocessing ---
def preprocess_data(df):
    """Separates features and target, and scales features."""
    print("\nPreprocessing data...")
    # Define target and features based on the anemia dataset
    target_column = 'Anemia_Severity'
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if features are all numeric
    if not all(X.dtypes.apply(pd.api.types.is_numeric_dtype)):
        print("Warning: Non-numeric features detected. Ensure all features are numeric before scaling.")
        # Handle non-numeric features here if necessary (e.g., encoding)
        # For this synthetic dataset, they should all be numeric.

    print(f"Features (X shape): {X.shape}")
    print(f"Target (y shape): {y.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled using StandardScaler.")

    return X_scaled, y, scaler, X.columns # Return feature names as well

# --- 3. Model Training ---
def train_classifiers(X_train, y_train):
    """Trains multiple classifiers."""
    print("\nTraining classifiers...")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5), # Specify n_neighbors
        "Naive Bayes": GaussianNB(),
        # Ensure SVC can handle multi-class (default is OvR) and enable probability
        "SVM": SVC(probability=True, kernel="rbf", random_state=42)
    }
    trained_models = {}
    feature_importances = {} # To store feature importances for RF

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} trained.")
        # Get feature importances if it's Random Forest
        if name == "Random Forest":
            if hasattr(model, 'feature_importances_'):
                feature_importances[name] = model.feature_importances_

    return trained_models, feature_importances

# --- 4. Model Evaluation ---
def evaluate_model(name, model, X_test, y_test, n_classes, class_labels):
    """Evaluates a single model and plots confusion matrix and an aggregate ROC curve
       using a simple plot style."""
    print(f"\n--- Evaluating Model: {name} ---")
    y_pred = model.predict(X_test)

    # Ensure predict_proba is available for ROC
    if not hasattr(model, "predict_proba"):
        print(f"Warning: Model {name} does not have predict_proba. Skipping ROC curve plot.")
        # Calculate and print metrics/plots that don't need probabilities
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_labels))
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        # Confusion matrix heatmap color - set to "Oranges"
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
        return accuracy # Return accuracy even without ROC

    y_prob = model.predict_proba(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Classification Report
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_labels))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    # Confusion matrix heatmap color - set to "Oranges"
    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # --- Aggregate ROC Curve (Macro-average using simplified plot style) ---
    # Calculate ROC curves for each class (One-vs-Rest)
    # Then compute the macro-average data to represent the overall curve
    lb = LabelBinarizer()
    y_test_bin = lb.fit_transform(y_test)
    # n_classes is already available from the main block or can be derived from y_test_bin.shape[1]

    fpr_per_class = dict()
    tpr_per_class = dict()
    roc_auc_per_class = dict()
    for i in range(n_classes):
        fpr_per_class[i], tpr_per_class[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc_per_class[i] = auc(fpr_per_class[i], tpr_per_class[i])

    # Compute macro-average ROC curve data points
    all_fpr = np.unique(np.concatenate([fpr_per_class[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_per_class[i], tpr_per_class[i])
    mean_tpr /= n_classes

    # Assign the calculated macro-average data to simple variable names
    # to match the requested plotting code syntax
    fpr = all_fpr
    tpr = mean_tpr
    roc_auc = auc(fpr, tpr) # Calculate AUC for the macro curve

    # Plot using the exact style you requested
    plt.figure(figsize=(8, 6))
    # ROC curve line color - set to 'red'
    plt.plot(fpr, tpr, color='red', label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"{name} - ROC Curve") # Use the exact requested title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True) # Ensure grid is on
    plt.show()

    return accuracy # Return accuracy for comparison

# --- 5. Save Model ---
def save_best_model(model, scaler, feature_names, filename="anemia_best_model.pkl"):
    """Saves the best performing model and the scaler."""
    print(f"\nSaving model and scaler to {filename}...")
    # Include feature names used during training for consistency
    with open(filename, "wb") as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': feature_names}, f)
    print("Model and scaler saved successfully.")

# --- 6. Visualizations ---
def plot_visualizations(df, feature_names, target_column, class_labels_map):
    """Generates various data visualizations."""
    print("\nGenerating visualizations...")

    # Map numeric target to labels for plotting
    df_plot = df.copy()
    df_plot['Severity_Label'] = df_plot[target_column].map(class_labels_map)


    # --- Scatter Plot (Example: HGB vs MCV) ---
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="HGB",
        y="MCV",
        hue="Severity_Label", # Use mapped labels
        style="Severity_Label", # Use mapped labels
        palette="viridis", # Use a different palette suitable for multi-class
        data=df_plot
    )
    plt.title("Scatter Plot: Hemoglobin (HGB) vs Mean Corpuscular Volume (MCV)")
    plt.xlabel("Hemoglobin (g/dL)")
    plt.ylabel("MCV (fL)")
    plt.grid(True)
    plt.legend(title="Anemia Severity")
    plt.show()

    # --- Pair Plot (Key Features) ---
    # Select key features most indicative of anemia
    key_features = ["HGB", "HCT", "MCV", "RDW", "Severity_Label"]
    print(f"\nGenerating Pair Plot for features: {key_features[:-1]}...") # Exclude label from print
    sns.pairplot(df_plot[key_features], hue="Severity_Label", palette="viridis")
    plt.suptitle("Pairplot of Key Anemia Features vs Severity", y=1.02)
    plt.show()

    # --- 3D Scatter Plot (Example: HGB, MCV, RDW) ---
    print("\nGenerating 3D Scatter Plot (HGB, MCV, RDW)...")
    fig = plt.figure(figsize=(10, 8))
    # Ensure Axes3D is imported
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for each class explicitly
    colors = df_plot['Severity_Label'].map({
        'Normal': 'green',
        'Mild': 'blue',
        'Moderate': 'orange',
        'Severe': 'red'
    })

    scatter = ax.scatter(
        df_plot["HGB"], df_plot["MCV"], df_plot["RDW"],
        c=colors, # Use mapped colors
        s=60, alpha=0.7, marker='o'
    )

    ax.set_xlabel("Hemoglobin (HGB - g/dL)")
    ax.set_ylabel("MCV (fL)")
    ax.set_zlabel("RDW (%)")
    ax.set_title("3D Scatter Plot: HGB vs MCV vs RDW")

    # Create a custom legend (handles might be complex with direct color mapping)
    # Simple approach: Create proxy artists
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Normal', markerfacecolor='green', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='Mild', markerfacecolor='blue', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='Moderate', markerfacecolor='orange', markersize=10),
                      Line2D([0], [0], marker='o', color='w', label='Severe', markerfacecolor='red', markersize=10)]
    ax.legend(handles=legend_elements, title="Anemia Severity")

    plt.show()

    # --- Histograms of Features (using pandas .hist()) ---
    print("\nGenerating Histograms of Features...")
    # Plot histograms for all numeric columns using pandas .hist()
    # Adjust layout to prevent suptitle overlap
    df.hist(figsize=(14, 10), bins=15, edgecolor='black', color='skyblue')
    plt.suptitle("Histograms of Features", y=1.01)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to prevent suptitle overlap
    plt.show()


    # --- Box Plot of Features ---
    print("\nGenerating Box Plot of Features...")
    plt.figure(figsize=(12, 6))
    # Melt the dataframe for easier plotting with seaborn
    df_melt = pd.melt(df, id_vars=target_column, var_name='Feature', value_name='Value')
    sns.boxplot(x='Feature', y='Value', data=df_melt, palette="tab10")
    # sns.boxplot(data=df.drop(columns=[target_column])) # Alternative simpler boxplot
    plt.xticks(rotation=45)
    plt.title("Boxplot of Features")
    plt.tight_layout() # Adjust layout
    plt.show()

    # --- Correlation Heatmap ---
    print("\nGenerating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    # Calculate correlation on the original numeric dataframe
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation Between Features and Target")
    plt.tight_layout() # Adjust layout
    plt.show()

# --- 7. Feature Importance Plot (for Random Forest) ---
def plot_feature_importance(importance, names, model_name):
    """Plots feature importance for a given model."""
    if importance is None or len(importance) == 0:
        print(f"No feature importance data available for {model_name}.")
        return

    print(f"\nGenerating Feature Importance plot for {model_name}...")
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create DataFrame for plotting
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(f'{model_name} - Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Deep Researchs') # Updated y-axis label
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    # Define dataset path and class labels
    # Updated dataset path
    data_path = "anemia_severity_dataset_balanced_auc.csv"
    # Define class labels corresponding to numeric values 0, 1, 2, 3
    class_labels_map = {0: 'Normal', 1: 'Mild', 2: 'Moderate', 3: 'Severe'}
    class_labels_list = ['Normal', 'Mild', 'Moderate', 'Severe'] # For metrics target_names
    n_classes = len(class_labels_list)

    # --- Workflow ---
    # 1. Load Data
    df = load_data(data_path)

    # 2. Preprocess Data
    X_scaled, y, scaler, feature_names = preprocess_data(df)

    # 3. Split Data
    # Use 25% test size and stratify for balanced class representation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
    print(f"\nData split into Train/Test sets: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    print("Test set target distribution:\n", pd.Series(y_test).value_counts(normalize=True))


    # 4. Train Models
    trained_models, rf_feature_importances = train_classifiers(X_train, y_train)

    # 5. Evaluate Models
    model_accuracies = {}
    for name, model in trained_models.items():
        # Pass n_classes and class_labels_list to evaluate_model
        accuracy = evaluate_model(name, model, X_test, y_test, n_classes=n_classes, class_labels=class_labels_list)
        model_accuracies[name] = accuracy

    # --- Summary and Best Model Selection ---
    print("\n--- Model Performance Summary ---")
    best_model_name = ""
    best_accuracy = 0
    for name, acc in model_accuracies.items():
        print(f"{name}: Accuracy = {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

    print(f"\nBest performing model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

    # 6. Save the Best Model (Example: Saving the one with highest accuracy)
    if best_model_name:
        # Save best model including scaler and feature names
        save_best_model(trained_models[best_model_name], scaler, feature_names, filename=f"anemia_{best_model_name.replace(' ', '_')}_best_model.pkl")
    else:
        print("Could not determine the best model to save.")

    # 7. Visualizations on the original dataframe (before scaling)
    # Pass all required arguments to plot_visualizations
    plot_visualizations(df, feature_names, 'Anemia_Severity', class_labels_map)

    # 8. Plot Feature Importance (Only for Random Forest in this example)
    if "Random Forest" in rf_feature_importances:
        # Pass feature names for the plot labels
        plot_feature_importance(rf_feature_importances["Random Forest"], feature_names, "Random Forest")

    print("\n--- Analysis Complete ---")
