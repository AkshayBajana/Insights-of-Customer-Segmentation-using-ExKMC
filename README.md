# Explainable Customer Segmentation

This application implements customer segmentation using K-means clustering with explainability through decision trees. It provides an interactive interface for analyzing customer data and understanding cluster assignments.

## Features

- Interactive data upload and feature selection
- Automatic optimal cluster number detection using:
  - Elbow Method
  - Silhouette Score
- Cluster visualization and analysis
- Decision tree-based cluster explanation
- Downloadable results

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:
```bash
streamlit run app.py
```

## Usage

1. Prepare your customer data in CSV format
2. Upload the CSV file through the application interface
3. Select the features you want to use for clustering
4. Review the optimal number of clusters using the elbow curve and silhouette score
5. Choose the number of clusters and perform the segmentation
6. Explore the results and explanations
7. Download the segmented data if needed

## Data Format

Your CSV file should contain customer data with numerical features. Example features might include:
- Purchase frequency
- Average order value
- Customer lifetime value
- Recency of purchase
- etc.

## Output

The application provides:
- Cluster assignments for each customer
- Cluster distribution visualization
- Cluster characteristics
- Decision tree visualization
- Feature importance analysis
- Downloadable results in CSV format 