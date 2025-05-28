import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler

def find_optimal_clusters(X, max_clusters):
    """
    Find optimal number of clusters using elbow method and silhouette score.
    
    Args:
        X: Scaled feature matrix
        max_clusters: Maximum number of clusters to try
        
    Returns:
        elbow_data: List of inertia values
        silhouette_data: List of silhouette scores
    """
    elbow_data = []
    silhouette_data = []
    
    # Calculate inertia for elbow method
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        elbow_data.append(kmeans.inertia_)
        
        # Calculate silhouette score for k >= 2
        if k >= 2:
            silhouette_data.append(silhouette_score(X, kmeans.labels_))
    
    return elbow_data, silhouette_data

def exkmc(X, n_clusters, max_depth=5, random_state=42):
    """
    Implement ExKMC (Explainable K-Means Clustering) algorithm.
    
    Args:
        X: Feature matrix
        n_clusters: Number of clusters
        max_depth: Maximum depth of the decision tree
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary containing clustering results and explanations
    """
    # Step 1: Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(X)
    
    # Step 2: Train a decision tree to explain the clusters
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X, clusters)
    
    # Step 3: Get feature importances
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': tree.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Step 4: Calculate cluster centers and characteristics
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=X.columns
    )
    
    # Step 5: Generate cluster explanations
    explanations = []
    for i in range(n_clusters):
        # Get samples in this cluster
        cluster_samples = X[clusters == i]
        
        # Calculate mean values for each feature
        cluster_means = cluster_samples.mean()
        
        # Get top features for this cluster
        top_features = importances.head(3)
        
        # Generate explanation
        explanation = f"Cluster {i}:\n"
        explanation += "Top distinguishing features:\n"
        for _, row in top_features.iterrows():
            feature = row['Feature']
            value = cluster_means[feature]
            explanation += f"- {feature}: {value:.2f}\n"
        explanations.append(explanation)
    
    return {
        'clusters': clusters,
        'tree': tree,
        'importances': importances,
        'cluster_centers': cluster_centers,
        'explanations': explanations,
        'kmeans': kmeans
    }

def explain_clusters(X, clusters):
    """
    Explain clusters using ExKMC algorithm.
    
    Args:
        X: Feature matrix
        clusters: Cluster assignments
        
    Returns:
        Dictionary containing the decision tree and explanation text
    """
    n_clusters = len(np.unique(clusters))
    result = exkmc(X, n_clusters)
    
    # Generate overall explanation text
    explanation = "Cluster Explanation:\n\n"
    for cluster_explanation in result['explanations']:
        explanation += cluster_explanation + "\n"
    
    return {
        'tree': result['tree'],
        'explanation': explanation,
        'importances': result['importances'],
        'cluster_centers': result['cluster_centers']
    } 