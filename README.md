# Fraud Detection with PyOD AutoEncoder

## Project Overview

This project implements a comprehensive fraud detection system using AutoEncoder from the PyOD library to detect anomalies in credit card transactions. The AutoEncoder learns to reconstruct normal transactions and identifies transactions with high reconstruction errors as potential fraud.

## Requirements

### System Requirements
- Python 3.7 or higher
- Git (for version control)
- Visual Studio Code (recommended IDE)

### Python Dependencies
Create a `requirements.txt` file with the following dependencies:

```
pyod>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tensorflow>=2.8.0
keras>=2.8.0
```

## Installation Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/shaiksameer11/MSCS-633_Assignment_4.git
cd MSCS-633_Assignment_4
```

### 2. Set up Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv fraud_detection_env

# Activate virtual environment
# On macOS/Linux:
source fraud_detection_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install PyOD
```bash
# PyOD can also be installed separately if needed
pip install pyod
```

## Project Structure

```
fraud-detection-pyod/
│
├── fraud_detection.py          # Main implementation file
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── data/                       # Data directory (if using real dataset)
│   └── creditcard.csv         # Kaggle credit card dataset (optional)
├── results/                    # Output directory
│   ├── fraud_detection_results.csv
│   └── model_visualizations.png
└── notebooks/                  # Jupyter notebooks (optional)
    └── experiment_analysis.ipynb
```

## Dataset Information

### Option 1: Sample Dataset (Default)
The script generates a synthetic dataset with:
- 9,000 normal transactions
- 1,000 fraudulent transactions
- 10 PCA-transformed features (V1-V10)
- Amount and Time features

### Option 2: Kaggle Dataset
To use the real Kaggle credit card fraud dataset:
1. Download from: https://www.kaggle.com/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` in the `data/` directory
3. Modify the main function to use the real dataset:

```python
fraud_detector.run_complete_analysis('data/creditcard.csv')
```

## Usage

### Basic Usage
```bash
python fraud_detection.py
```

### Advanced Usage
```python
from fraud_detection import FraudDetectionSystem

# Initialize with custom parameters
detector = FraudDetectionSystem(
    contamination=0.05,  # Expected fraud rate
    random_state=42
)

# Run with real dataset
detector.run_complete_analysis('data/creditcard.csv')
```

## Model Configuration

### AutoEncoder Architecture
- **Input Layer**: Number of features in dataset
- **Hidden Layers**: [32, 16, 8, 16, 32] (encoder-decoder structure)
- **Training Epochs**: 100
- **Batch Size**: 32
- **Contamination Rate**: 10% (adjustable)

### Key Parameters
- `contamination`: Expected proportion of outliers (fraud cases)
- `hidden_neurons`: Architecture of the neural network
- `epochs`: Number of training iterations
- `batch_size`: Size of training batches

## Output and Results

### Generated Files
1. **fraud_detection_results.csv**: Detailed predictions and scores
2. **Visualization plots**: ROC curve, confusion matrix, score distributions
3. **Console output**: Performance metrics and statistics

### Performance Metrics
- Classification Report (Precision, Recall, F1-score)
- ROC AUC Score
- Confusion Matrix
- Reconstruction Error Analysis

## Troubleshooting

### Common Issues

1. **TensorFlow Installation Issues**
   ```bash
   pip install tensorflow==2.8.0
   ```

2. **PyOD Import Errors**
   ```bash
   pip install --upgrade pyod
   ```

3. **Memory Issues with Large Datasets**
    - Reduce batch size
    - Use data sampling for initial experiments

### Performance Optimization
- Adjust `hidden_neurons` for different dataset sizes
- Tune `contamination` parameter based on actual fraud rate
- Experiment with different `epochs` and `batch_size`

## Best Practices

1. **Data Preprocessing**
    - Always scale/normalize features
    - Handle missing values appropriately
    - Consider feature engineering for better performance

2. **Model Validation**
    - Use stratified train-test split
    - Consider cross-validation for robust evaluation
    - Monitor for overfitting

3. **Production Deployment**
    - Implement model versioning
    - Set up monitoring for model performance
    - Regular retraining with new data

## References

- PyOD Documentation: https://pyod.readthedocs.io/
- Kaggle Credit Card Fraud Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
- AutoEncoder for Anomaly Detection: https://keras.io/examples/timeseries/timeseries_anomaly_detection/
