"""
Fraud Detection System using Multiple PyOD Models
=================================================

This script implements a comprehensive fraud detection system using multiple anomaly detection
algorithms from PyOD, including a custom AutoEncoder implementation using Keras/TensorFlow
to avoid PyOD AutoEncoder API issues.

Author: AI Assistant
Date: August 2025
Dependencies: pyod, pandas, numpy, scikit-learn, matplotlib, seaborn, tensorflow
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Try to import TensorFlow for custom AutoEncoder
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    print("TensorFlow available - will use custom AutoEncoder implementation")
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available - will use Isolation Forest only")

class CustomAutoEncoder:
    """
    Custom AutoEncoder implementation using TensorFlow/Keras for fraud detection.
    """

    def __init__(self, input_dim, contamination=0.1, epochs=50, batch_size=32, random_state=42):
        self.input_dim = input_dim
        self.contamination = contamination
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.encoder = None
        self.threshold = None

        # Set random seeds
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

    def _build_autoencoder(self):
        """Build the autoencoder architecture."""
        # Input layer
        input_layer = keras.Input(shape=(self.input_dim,))

        # Encoder
        encoded = layers.Dense(32, activation='relu')(input_layer)
        encoded = layers.Dense(16, activation='relu')(encoded)
        encoded = layers.Dense(8, activation='relu')(encoded)

        # Decoder
        decoded = layers.Dense(16, activation='relu')(encoded)
        decoded = layers.Dense(32, activation='relu')(decoded)
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)

        # Build models
        self.model = keras.Model(input_layer, decoded)
        self.encoder = keras.Model(input_layer, encoded)

        # Compile
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, X):
        """Fit the autoencoder to the data."""
        self._build_autoencoder()

        # Train the autoencoder
        history = self.model.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=1,
            shuffle=True
        )

        # Calculate reconstruction errors for threshold
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)

        # Set threshold based on contamination rate
        self.threshold = np.percentile(mse, 100 * (1 - self.contamination))

        return history

    def predict(self, X):
        """Predict anomalies (0 = normal, 1 = anomaly)."""
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return (mse > self.threshold).astype(int)

    def decision_function(self, X):
        """Return anomaly scores (reconstruction errors)."""
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        return mse

class FraudDetectionSystem:
    """
    A comprehensive fraud detection system using multiple anomaly detection algorithms.
    """

    def __init__(self, contamination=0.1, random_state=42, use_autoencoder=True):
        """
        Initialize the fraud detection system.

        Args:
            contamination (float): The proportion of outliers in the dataset
            random_state (int): Random state for reproducibility
            use_autoencoder (bool): Whether to use AutoEncoder (requires TensorFlow)
        """
        self.contamination = contamination
        self.random_state = random_state
        self.use_autoencoder = use_autoencoder and TF_AVAILABLE
        self.scaler = StandardScaler()
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # Create results directory
        self.results_dir = "results"
        self._create_results_directory()

        # Generate timestamp for file naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def _create_results_directory(self):
        """Create results directory if it doesn't exist."""
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            print(f"Created results directory: {self.results_dir}/")
        else:
            print(f"Using existing results directory: {self.results_dir}/")

    def load_and_prepare_data(self, file_path=None, use_sample=True):
        """
        Load and prepare the credit card dataset.

        Args:
            file_path (str): Path to the dataset file
            use_sample (bool): If True, creates a sample dataset for demonstration

        Returns:
            pd.DataFrame: Prepared dataset
        """
        if use_sample or file_path is None:
            # Create a sample dataset for demonstration
            print("Creating sample credit card transaction dataset...")
            np.random.seed(self.random_state)

            # Generate normal transactions
            n_normal = 9000
            normal_data = {
                'Amount': np.random.lognormal(3, 1, n_normal),
                'Time': np.random.uniform(0, 172800, n_normal),
                'V1': np.random.normal(0, 1, n_normal),
                'V2': np.random.normal(0, 1, n_normal),
                'V3': np.random.normal(0, 1, n_normal),
                'V4': np.random.normal(0, 1, n_normal),
                'V5': np.random.normal(0, 1, n_normal),
                'V6': np.random.normal(0, 1, n_normal),
                'V7': np.random.normal(0, 1, n_normal),
                'V8': np.random.normal(0, 1, n_normal),
                'V9': np.random.normal(0, 1, n_normal),
                'V10': np.random.normal(0, 1, n_normal)
            }

            # Generate fraudulent transactions
            n_fraud = 1000
            fraud_data = {
                'Amount': np.random.lognormal(5, 2, n_fraud),
                'Time': np.random.uniform(0, 172800, n_fraud),
                'V1': np.random.normal(2, 1.5, n_fraud),
                'V2': np.random.normal(-1, 2, n_fraud),
                'V3': np.random.normal(3, 1, n_fraud),
                'V4': np.random.normal(-2, 1.5, n_fraud),
                'V5': np.random.normal(1.5, 2, n_fraud),
                'V6': np.random.normal(-3, 1, n_fraud),
                'V7': np.random.normal(2.5, 1.5, n_fraud),
                'V8': np.random.normal(-1.5, 2, n_fraud),
                'V9': np.random.normal(4, 1, n_fraud),
                'V10': np.random.normal(-2.5, 1.5, n_fraud)
            }

            # Create DataFrames
            normal_df = pd.DataFrame(normal_data)
            normal_df['Class'] = 0

            fraud_df = pd.DataFrame(fraud_data)
            fraud_df['Class'] = 1

            # Combine datasets
            df = pd.concat([normal_df, fraud_df], ignore_index=True)

        else:
            # Load from file
            print(f"Loading dataset from {file_path}...")
            df = pd.read_csv(file_path)

        # Shuffle the dataset
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        print(f"Dataset shape: {df.shape}")
        print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.2f}%)")
        print(f"Normal cases: {(df['Class'] == 0).sum()} ({(df['Class'] == 0).sum()/len(df)*100:.2f}%)")

        return df

    def preprocess_data(self, df, test_size=0.2):
        """
        Preprocess the data for training.
        """
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Scale the features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print(f"Training set shape: {self.X_train_scaled.shape}")
        print(f"Test set shape: {self.X_test_scaled.shape}")

    def train_models(self):
        """
        Train multiple anomaly detection models.
        """
        print("Training anomaly detection models...")

        # 1. Isolation Forest
        print("Training Isolation Forest...")
        self.models['IsolationForest'] = IForest(
            contamination=self.contamination,
            n_estimators=100,
            random_state=self.random_state
        )
        self.models['IsolationForest'].fit(self.X_train_scaled)

        # 2. Local Outlier Factor
        print("Training Local Outlier Factor...")
        self.models['LOF'] = LOF(
            contamination=self.contamination,
            n_neighbors=20
        )
        self.models['LOF'].fit(self.X_train_scaled)

        # 3. Custom AutoEncoder (if TensorFlow is available)
        if self.use_autoencoder:
            print("Training Custom AutoEncoder...")
            self.models['AutoEncoder'] = CustomAutoEncoder(
                input_dim=self.X_train_scaled.shape[1],
                contamination=self.contamination,
                epochs=50,
                batch_size=32,
                random_state=self.random_state
            )
            self.models['AutoEncoder'].fit(self.X_train_scaled)

        print("All models trained successfully!")

    def evaluate_models(self):
        """
        Evaluate all trained models.
        """
        results = {}

        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)

        for model_name, model in self.models.items():
            print(f"\n--- {model_name} ---")

            # Make predictions
            y_pred = model.predict(self.X_test_scaled)
            anomaly_scores = model.decision_function(self.X_test_scaled)

            # Calculate metrics
            roc_auc = roc_auc_score(self.y_test, anomaly_scores)
            cm = confusion_matrix(self.y_test, y_pred)

            # Store results
            results[model_name] = {
                'predictions': y_pred,
                'scores': anomaly_scores,
                'roc_auc': roc_auc,
                'confusion_matrix': cm
            }

            # Print results
            print(f"ROC AUC Score: {roc_auc:.4f}")
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))
            print("Confusion Matrix:")
            print(cm)

        return results

    def create_visualizations(self, results):
        """
        Create comprehensive visualizations for all models.
        """
        n_models = len(results)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))

        if n_models == 1:
            axes = axes.reshape(2, 1)

        for i, (model_name, result) in enumerate(results.items()):
            # ROC Curves
            fpr, tpr, _ = roc_curve(self.y_test, result['scores'])
            axes[0, i].plot(fpr, tpr, color='darkorange', lw=2,
                           label=f'ROC curve (AUC = {result["roc_auc"]:.4f})')
            axes[0, i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[0, i].set_xlim([0.0, 1.0])
            axes[0, i].set_ylim([0.0, 1.05])
            axes[0, i].set_xlabel('False Positive Rate')
            axes[0, i].set_ylabel('True Positive Rate')
            axes[0, i].set_title(f'{model_name} - ROC Curve')
            axes[0, i].legend(loc="lower right")
            axes[0, i].grid(True, alpha=0.3)

            # Confusion Matrix
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Normal', 'Fraud'],
                       yticklabels=['Normal', 'Fraud'], ax=axes[1, i])
            axes[1, i].set_title(f'{model_name} - Confusion Matrix')
            axes[1, i].set_ylabel('True Label')
            axes[1, i].set_xlabel('Predicted Label')

        plt.tight_layout()

        # Save the figure
        figure_filename = f"fraud_detection_analysis_{self.timestamp}.png"
        figure_path = os.path.join(self.results_dir, figure_filename)
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {figure_path}")

        # Show the plot
        plt.show()

        # Create additional detailed visualizations
        self._create_detailed_visualizations(results)

        # Summary comparison
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        for model_name, result in results.items():
            print(f"{model_name:15s}: ROC AUC = {result['roc_auc']:.4f}")

    def _create_detailed_visualizations(self, results):
        """
        Create additional detailed visualizations.
        """
        # Best model detailed analysis
        best_model = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        best_result = results[best_model]

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Detailed Analysis - Best Model: {best_model}', fontsize=16)

        # 1. Anomaly Score Distribution
        axes[0, 0].hist(best_result['scores'][self.y_test == 0], bins=50, alpha=0.7,
                       label='Normal', color='blue', density=True)
        axes[0, 0].hist(best_result['scores'][self.y_test == 1], bins=50, alpha=0.7,
                       label='Fraud', color='red', density=True)
        axes[0, 0].set_xlabel('Anomaly Score')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Distribution of Anomaly Scores')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. ROC Curve for best model
        fpr, tpr, _ = roc_curve(self.y_test, best_result['scores'])
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC curve (AUC = {best_result["roc_auc"]:.4f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'{best_model} - ROC Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(self.y_test, best_result['scores'])
        ap_score = average_precision_score(self.y_test, best_result['scores'])

        axes[1, 0].plot(recall, precision, color='green', lw=2,
                       label=f'PR curve (AP = {ap_score:.4f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Feature importance or score scatter
        axes[1, 1].scatter(range(len(best_result['scores'])),
                          best_result['scores'],
                          c=self.y_test, cmap='coolwarm', alpha=0.6)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Anomaly Score')
        axes[1, 1].set_title('Anomaly Scores by Sample')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save detailed analysis
        detailed_filename = f"detailed_analysis_{best_model}_{self.timestamp}.png"
        detailed_path = os.path.join(self.results_dir, detailed_filename)
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        print(f"Detailed analysis saved to: {detailed_path}")

        plt.show()

    def save_results(self, results, output_file=None):
        """
        Save results to CSV file in the results directory.
        """
        if output_file is None:
            output_file = f'fraud_detection_results_{self.timestamp}.csv'

        # Ensure file is saved in results directory
        output_path = os.path.join(self.results_dir, output_file)

        # Use the best performing model (highest ROC AUC)
        best_model = max(results.keys(), key=lambda k: results[k]['roc_auc'])
        best_result = results[best_model]

        results_df = pd.DataFrame({
            'True_Label': self.y_test,
            'Predicted_Label': best_result['predictions'],
            'Anomaly_Score': best_result['scores'],
            'Best_Model': best_model,
            'Is_Correctly_Classified': self.y_test == best_result['predictions']
        })

        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path} (using best model: {best_model})")

        # Save model comparison summary
        summary_file = f'model_comparison_summary_{self.timestamp}.csv'
        summary_path = os.path.join(self.results_dir, summary_file)

        summary_data = []
        for model_name, result in results.items():
            # Calculate additional metrics
            tn, fp, fn, tp = result['confusion_matrix'].ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            summary_data.append({
                'Model': model_name,
                'ROC_AUC': result['roc_auc'],
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1_score,
                'True_Positives': tp,
                'True_Negatives': tn,
                'False_Positives': fp,
                'False_Negatives': fn
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_path, index=False)
        print(f"Model comparison summary saved to: {summary_path}")

        return output_path, summary_path

    def run_complete_analysis(self, file_path=None):
        """
        Run the complete fraud detection analysis.
        """
        print("Starting Comprehensive Fraud Detection Analysis")
        print("="*50)

        # Load and prepare data
        df = self.load_and_prepare_data(file_path)

        # Preprocess data
        self.preprocess_data(df)

        # Train models
        self.train_models()

        # Evaluate models
        results = self.evaluate_models()

        # Create visualizations
        self.create_visualizations(results)

        # Save results
        results_path, summary_path = self.save_results(results)

        print("\n" + "="*50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Generated files in '{self.results_dir}/' directory:")
        print(f"  - CSV Results: {os.path.basename(results_path)}")
        print(f"  - Summary: {os.path.basename(summary_path)}")
        print(f"  - Main Visualization: fraud_detection_analysis_{self.timestamp}.png")
        print(f"  - Detailed Analysis: detailed_analysis_[best_model]_{self.timestamp}.png")
        print("="*50)

def main():
    """
    Main function to run the fraud detection system.
    """
    # Initialize the fraud detection system
    fraud_detector = FraudDetectionSystem(
        contamination=0.05,
        random_state=42,
        use_autoencoder=True  # Set to False if TensorFlow issues persist
    )

    # Run the complete analysis
    fraud_detector.run_complete_analysis('data/creditcard.csv')  # Change to None for sample data

if __name__ == "__main__":
    main()