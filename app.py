import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
from sklearn.datasets import make_blobs
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import exkmc
import warnings
import io
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Insights of Customer Segmentation using K-Means", layout="wide")

st.title("Insights of Customer Segmentation using K-Means")
st.write("""
This application performs K-means clustering and provides interpretable cluster assignments 
through decision trees, making it easy to understand how customers are segmented.
""")

# Add option to use synthetic data
use_synthetic = st.checkbox("Use Synthetic Data", value=False, key="use_synthetic")

if use_synthetic:
    # Parameters for synthetic data
    st.subheader("Synthetic Data Parameters")
    n_samples = st.slider("Number of samples", 100, 1000, 1000, key="synthetic_samples")
    n_features = st.slider("Number of features", 2, 5, 2, key="synthetic_features")
    n_clusters = st.slider("Number of clusters", 2, 5, 3, key="synthetic_clusters")
    
    # Generate synthetic data
    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=2.5,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data['True_Cluster'] = true_labels
    
    # Display data preview
    st.subheader("Synthetic Data Preview")
    st.dataframe(data.head())
    
    # Visualize the data
    if n_features == 2:
        st.subheader("Data Visualization")
        fig = px.scatter(
            data,
            x='Feature_1',
            y='Feature_2',
            color='True_Cluster',
            title='Synthetic Data with True Clusters'
        )
        st.plotly_chart(fig)
    
    # Prepare data for clustering
    X = data[feature_names].copy()
    
else:
    # File uploader for real data
    uploaded_file = st.file_uploader("Upload your customer data (CSV or Excel file)", type=['csv', 'xlsx', 'xls'], key="file_uploader")
    
    if uploaded_file is not None:
        try:
            # Read the data based on file type
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:  # Excel file
                data = pd.read_excel(uploaded_file)
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Check for missing values
            missing_values = data.isnull().sum()
            if missing_values.any():
                st.warning("⚠️ Missing values detected in the data. They will be handled automatically.")
                st.write("Missing values per column:")
                st.dataframe(missing_values[missing_values > 0])
            
            # Only allow numeric columns for clustering
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("Error: The dataset must contain at least 2 numeric columns for clustering.")
                st.stop()
            
            # Feature selection
            st.subheader("Feature Selection")
            features = st.multiselect(
                "Select features for clustering (numeric only)",
                options=numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,
                key="feature_selector"
            )
            
            if len(features) >= 2:
                X = data[features].copy()
            else:
                st.warning("Please select at least 2 features for clustering.")
                st.stop()
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            st.stop()
    else:
        st.warning("Please upload a CSV or Excel file or use synthetic data.")
        st.stop()

# Handle missing values
try:
    # Check for missing values
    if X.isnull().any().any():
        st.info("Handling missing values using mean imputation...")
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        X = X_imputed
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Elbow Method to find optimal number of clusters
    st.subheader("Elbow Method Analysis")
    st.write("""
    The Elbow Method helps us determine the optimal number of clusters by analyzing the Within-Cluster Sum of Squares (WCSS).
    The 'elbow' point in the graph indicates where adding more clusters doesn't significantly reduce WCSS.
    """)
    
    # Calculate WCSS for different numbers of clusters
    max_clusters = min(10, len(data) - 1)
    wcss = []
    k_range = range(1, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    
    # Create Elbow Method plot
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(k_range),
        y=wcss,
        mode='lines+markers',
        name='WCSS'
    ))
    
    fig_elbow.update_layout(
        title='Elbow Method for Optimal K',
        xaxis_title='Number of Clusters (K)',
        yaxis_title='Within-Cluster Sum of Squares (WCSS)',
        showlegend=True
    )
    
    st.plotly_chart(fig_elbow)
    
    # Add explanation of the Elbow Method
    st.write("""
    ### Understanding the Elbow Method:
    1. **WCSS (Within-Cluster Sum of Squares)**: Measures the compactness of clusters
    2. **Elbow Point**: The point where the rate of decrease in WCSS sharply changes
    3. **Optimal K**: Usually found at the 'elbow' of the curve
    """)
    
    # Number of clusters selection
    n_clusters = st.slider("Select number of clusters", 2, max_clusters, 3, key="kmeans_n_clusters")
    
    if st.button("Generate Clusters", key="generate_clusters"):
        # Perform K-means clustering with k-means++ initialization
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Add clusters to original data
        data['Cluster'] = clusters
        
        # Create scatter plot visualization
        st.subheader("Cluster Visualization")
        
        # Create figure
        fig = go.Figure()
        
        # Define colors for clusters
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'orange', 'purple', 'brown', 'pink']
        
        # Add scatter traces for each cluster
        for cluster in range(n_clusters):
            cluster_data = data[data['Cluster'] == cluster]
            
            fig.add_trace(go.Scatter(
                x=cluster_data[X.columns[0]],
                y=cluster_data[X.columns[1]],
                mode='markers',
                name=f'Grupo {cluster + 1}',
                marker=dict(
                    size=10,
                    color=colors[cluster % len(colors)],
                    opacity=0.7
                )
            ))
        
        # Add centroids
        centroids = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=X.columns
        )
        
        fig.add_trace(go.Scatter(
            x=centroids[X.columns[0]],
            y=centroids[X.columns[1]],
            mode='markers',
            name='Centroides',
            marker=dict(
                size=15,
                color='black',
                symbol='triangle-up'
            )
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Clúster de clientes (k = {n_clusters})',
            xaxis_title=X.columns[0],
            yaxis_title=X.columns[1],
            showlegend=True,
            legend_title="Grupos"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display cluster statistics
        st.subheader("Cluster Statistics")
        cluster_stats = data.groupby('Cluster')[X.columns].agg(['mean', 'std', 'min', 'max']).round(2)
        st.dataframe(cluster_stats)
        
        # Display cluster sizes
        st.subheader("Cluster Sizes")
        cluster_sizes = data['Cluster'].value_counts().sort_index()
        cluster_sizes_df = pd.DataFrame({
            'Cluster': cluster_sizes.index,
            'Size': cluster_sizes.values,
            'Percentage': (cluster_sizes.values / len(data) * 100).round(2)
        })
        st.dataframe(cluster_sizes_df)
        
        # Add feature comparison across clusters
        st.subheader("Feature Comparison Across Clusters")
        
        # Calculate average values for each feature in each cluster
        avg_df = data.groupby('Cluster')[X.columns].mean().reset_index()
        avg_df = avg_df.rename(columns={'Cluster': 'cluster'})
        
        # Create subplots
        fig = make_subplots(
            rows=1, 
            cols=len(X.columns),
            subplot_titles=[f'Average {col}' for col in X.columns],
            shared_yaxes=True
        )
        
        # Add bar plots for each feature
        for i, feature in enumerate(X.columns, 1):
            fig.add_trace(
                go.Bar(
                    x=avg_df['cluster'],
                    y=avg_df[feature],
                    name=feature,
                    marker_color=colors[i-1] if i <= len(colors) else colors[0]
                ),
                row=1, col=i
            )
        
        # Update layout
        fig.update_layout(
            title_text="Feature Comparison Across Clusters",
            height=500,
            showlegend=False,
            title_x=0.5,
            title_font_size=20
        )
        
        # Update axes labels
        for i in range(1, len(X.columns) + 1):
            fig.update_xaxes(title_text="Cluster", row=1, col=i)
            fig.update_yaxes(title_text="Average Value", row=1, col=i)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add detailed feature comparison table
        st.write("### Detailed Feature Comparison")
        comparison_df = avg_df.copy()
        comparison_df = comparison_df.round(2)
        st.dataframe(comparison_df)
        
        # Add scatter plot analysis section
        st.subheader("Scatter Plot Analysis")
        
        # Create a 2x2 grid of scatter plots
        fig_scatter = make_subplots(
            rows=2, 
            cols=2,
            subplot_titles=[
                "Annual Income vs Spending Score",
                "Age vs Annual Income",
                "Age vs Spending Score",
                "Feature Comparison"
            ]
        )
        
        # Annual Income vs Spending Score
        for cluster in range(n_clusters):
            cluster_data = data[data['Cluster'] == cluster]
            fig_scatter.add_trace(
                go.Scatter(
                    x=cluster_data[X.columns[0]],  # Annual Income
                    y=cluster_data[X.columns[1]],  # Spending Score
                    mode='markers',
                    name=f'Cluster {cluster + 1}',
                    marker=dict(
                        size=10,
                        color=colors[cluster % len(colors)],
                        opacity=0.7
                    )
                ),
                row=1, col=1
            )
        
        # Add centroids for Annual Income vs Spending Score
        centroids = pd.DataFrame(
            scaler.inverse_transform(kmeans.cluster_centers_),
            columns=X.columns
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=centroids[X.columns[0]],
                y=centroids[X.columns[1]],
                mode='markers',
                name='Centroids',
                marker=dict(
                    size=15,
                    color='black',
                    symbol='triangle-up'
                )
            ),
            row=1, col=1
        )
        
        # Age vs Annual Income
        for cluster in range(n_clusters):
            cluster_data = data[data['Cluster'] == cluster]
            fig_scatter.add_trace(
                go.Scatter(
                    x=cluster_data[X.columns[2]],  # Age
                    y=cluster_data[X.columns[0]],  # Annual Income
                    mode='markers',
                    name=f'Cluster {cluster + 1}',
                    marker=dict(
                        size=10,
                        color=colors[cluster % len(colors)],
                        opacity=0.7
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Age vs Spending Score
        for cluster in range(n_clusters):
            cluster_data = data[data['Cluster'] == cluster]
            fig_scatter.add_trace(
                go.Scatter(
                    x=cluster_data[X.columns[2]],  # Age
                    y=cluster_data[X.columns[1]],  # Spending Score
                    mode='markers',
                    name=f'Cluster {cluster + 1}',
                    marker=dict(
                        size=10,
                        color=colors[cluster % len(colors)],
                        opacity=0.7
                    ),
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Feature Comparison (if more than 3 features)
        if len(X.columns) > 3:
            for cluster in range(n_clusters):
                cluster_data = data[data['Cluster'] == cluster]
                fig_scatter.add_trace(
                    go.Scatter(
                        x=cluster_data[X.columns[3]],  # Additional feature
                        y=cluster_data[X.columns[0]],  # Annual Income
                        mode='markers',
                        name=f'Cluster {cluster + 1}',
                        marker=dict(
                            size=10,
                            color=colors[cluster % len(colors)],
                            opacity=0.7
                        ),
                        showlegend=False
                    ),
                    row=2, col=2
                )
        
        # Update layout
        fig_scatter.update_layout(
            title_text="Customer Segmentation Analysis",
            height=800,
            showlegend=True,
            title_x=0.5,
            title_font_size=20
        )
        
        # Update axes labels
        fig_scatter.update_xaxes(title_text="Annual Income (k$)", row=1, col=1)
        fig_scatter.update_yaxes(title_text="Spending Score (1-100)", row=1, col=1)
        fig_scatter.update_xaxes(title_text="Age", row=1, col=2)
        fig_scatter.update_yaxes(title_text="Annual Income (k$)", row=1, col=2)
        fig_scatter.update_xaxes(title_text="Age", row=2, col=1)
        fig_scatter.update_yaxes(title_text="Spending Score (1-100)", row=2, col=1)
        if len(X.columns) > 3:
            fig_scatter.update_xaxes(title_text=X.columns[3], row=2, col=2)
            fig_scatter.update_yaxes(title_text="Annual Income (k$)", row=2, col=2)
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Add cluster analysis summary
        st.subheader("Cluster Analysis Summary")
        st.write("""
        ### Key Insights:
        1. **Annual Income vs Spending Score**:
           - Shows the relationship between customer income and spending behavior
           - Helps identify high-value customers and spending patterns
        
        2. **Age Analysis**:
           - Reveals age-based spending patterns
           - Helps understand demographic distribution
        
        3. **Cluster Characteristics**:
           - Each cluster represents a distinct customer segment
           - Centroids show the average characteristics of each segment
        """)
        
        # Add feature importance analysis
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(kmeans.cluster_centers_).mean(axis=0)
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            feature_importance,
            x='Feature',
            y='Importance',
            title='Feature Importance in Cluster Assignment'
        )
        st.plotly_chart(fig_importance)
        
        # Add K-means algorithm explanation
        st.subheader("K-means Algorithm Process")
        st.write("""
        The K-means algorithm follows these steps:
        
        1. **Initialization**: Centroids are initialized using the k-means++ method
        2. **Assignment**: Each point is assigned to the nearest centroid
        3. **Update**: Centroids are recalculated based on assigned points
        4. **Iteration**: Steps 2-3 are repeated until convergence
        
        The algorithm converged in {} iterations.
        """.format(kmeans.n_iter_))
        
        # Add AI-powered cluster explanation
        st.subheader("AI-Powered Cluster Analysis")
        
        def generate_cluster_profile(cluster_data, feature_names, cluster_id, all_data):
            """Generate a natural language profile for a cluster"""
            profile = []
            
            # Basic cluster information
            size = len(cluster_data)
            percentage = (size / len(all_data)) * 100
            profile.append(f"Cluster {cluster_id} represents {size} customers ({percentage:.1f}% of the total customer base).")
            
            # Feature analysis
            for feature in feature_names:
                cluster_mean = cluster_data[feature].mean()
                overall_mean = all_data[feature].mean()
                diff_percentage = ((cluster_mean - overall_mean) / overall_mean) * 100
                
                if abs(diff_percentage) > 10:  # Only mention significant differences
                    if diff_percentage > 0:
                        profile.append(f"This cluster has {abs(diff_percentage):.1f}% higher {feature} compared to the average customer.")
                    else:
                        profile.append(f"This cluster has {abs(diff_percentage):.1f}% lower {feature} compared to the average customer.")
            
            # Identify key differentiators
            std_devs = cluster_data[feature_names].std()
            most_variable = std_devs.idxmax()
            profile.append(f"The most variable feature in this cluster is {most_variable}, indicating diverse customer behavior in this aspect.")
            
            return " ".join(profile)

        def generate_cluster_comparison(clusters_data, feature_names):
            """Generate comparison between clusters"""
            comparison = []
            
            # Find the most distinctive features between clusters
            cluster_means = pd.DataFrame([data[feature_names].mean() for data in clusters_data])
            feature_variations = cluster_means.std()
            most_distinctive = feature_variations.nlargest(2).index
            
            comparison.append("Key differences between clusters:")
            for feature in most_distinctive:
                values = cluster_means[feature]
                min_cluster = values.idxmin()
                max_cluster = values.idxmax()
                comparison.append(f"- {feature}: Cluster {min_cluster} has the lowest value while Cluster {max_cluster} has the highest.")
            
            return "\n".join(comparison)

        def generate_business_insights(cluster_data, feature_names, cluster_id):
            """Generate business insights for a cluster"""
            insights = []
            
            # Calculate feature correlations
            correlations = cluster_data[feature_names].corr()
            
            # Find strongest correlations
            for i in range(len(feature_names)):
                for j in range(i+1, len(feature_names)):
                    corr = correlations.iloc[i,j]
                    if abs(corr) > 0.5:  # Only consider strong correlations
                        feature1 = feature_names[i]
                        feature2 = feature_names[j]
                        if corr > 0:
                            insights.append(f"Strong positive correlation ({corr:.2f}) between {feature1} and {feature2}.")
                        else:
                            insights.append(f"Strong negative correlation ({corr:.2f}) between {feature1} and {feature2}.")
            
            return insights

        # Generate and display AI-powered explanations
        st.write("### AI-Generated Cluster Profiles")
        
        # Store cluster data for comparison
        clusters_data = []
        
        for cluster in range(n_clusters):
            cluster_data = data[data['Cluster'] == cluster]
            clusters_data.append(cluster_data)
            
            st.write(f"#### Cluster {cluster} Profile")
            
            # Generate and display cluster profile
            profile = generate_cluster_profile(cluster_data, X.columns, cluster, data)
            st.write(profile)
            
            # Generate and display business insights
            insights = generate_business_insights(cluster_data, X.columns, cluster)
            if insights:
                st.write("**Business Insights:**")
                for insight in insights:
                    st.write(f"- {insight}")
            
            # Display cluster statistics with interpretation
            st.write("**Key Statistics:**")
            stats = cluster_data[X.columns].describe().round(2)
            st.dataframe(stats)
            
            # Visualize feature distributions
            st.write("**Feature Distributions:**")
            for feature in X.columns:
                fig = px.histogram(
                    cluster_data,
                    x=feature,
                    title=f'Distribution of {feature} in Cluster {cluster}'
                )
                st.plotly_chart(fig)
            
            st.write("---")
        
        # Generate and display cluster comparison
        st.write("### Cluster Comparison")
        comparison = generate_cluster_comparison(clusters_data, X.columns)
        st.write(comparison)
        
        # Add strategic recommendations
        st.write("### Strategic Recommendations")
        st.write("""
        Based on the cluster analysis, here are some strategic recommendations:
        
        1. **Targeted Marketing**:
           - Develop specific marketing strategies for each cluster
           - Customize messaging based on cluster characteristics
           - Focus on the most distinctive features of each cluster
        
        2. **Product Development**:
           - Identify opportunities for product customization
           - Develop features that appeal to specific clusters
           - Consider cluster-specific pricing strategies
        
        3. **Customer Service**:
           - Tailor customer service approaches to cluster needs
           - Develop cluster-specific support channels
           - Create targeted retention strategies
        """)
        
        # Add predictive insights
        st.write("### Predictive Insights")
        st.write("""
        The decision tree model can be used to:
        
        1. **Predict New Customer Segments**:
           - Quickly assign new customers to appropriate clusters
           - Identify potential high-value customers
           - Predict customer behavior patterns
        
        2. **Monitor Cluster Evolution**:
           - Track changes in cluster characteristics over time
           - Identify emerging customer segments
           - Adapt strategies based on cluster shifts
        """)
        
        # Add data point visualization section
        st.subheader("Data Point Visualization")
        
        try:
            # Create interactive scatter plot for all features
            st.write("### Interactive Data Point Analysis")
            
            # Ensure we have numeric columns for visualization
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) < 2:
                st.warning("Not enough numeric columns for visualization. Please ensure your data has at least 2 numeric columns.")
                st.stop()
            
            # Feature selection for visualization
            x_feature = st.selectbox("Select X-axis feature", numeric_columns, key="x_feature")
            y_feature = st.selectbox("Select Y-axis feature", [col for col in numeric_columns if col != x_feature], key="y_feature")
            
            # Create scatter plot with cluster information
            fig = go.Figure()
            
            for cluster in range(n_clusters):
                cluster_data = data[data['Cluster'] == cluster]
                
                if len(cluster_data) > 0:  # Only plot if cluster has data points
                    # Add scatter trace for each cluster
                    fig.add_trace(go.Scatter(
                        x=cluster_data[x_feature],
                        y=cluster_data[y_feature],
                        mode='markers',
                        name=f'Cluster {cluster}',
                        marker=dict(
                            size=10,
                            opacity=0.7
                        ),
                        hovertemplate=(
                            f"<b>Cluster {cluster}</b><br>" +
                            f"{x_feature}: %{{x:.2f}}<br>" +
                            f"{y_feature}: %{{y:.2f}}<br>" +
                            "<extra></extra>"
                        )
                    ))
            
            # Update layout
            fig.update_layout(
                title=f'Cluster Distribution: {x_feature} vs {y_feature}',
                xaxis_title=x_feature,
                yaxis_title=y_feature,
                hovermode='closest',
                showlegend=True,
                legend_title="Clusters"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add cluster statistics for selected features
            st.write("### Cluster Statistics for Selected Features")
            selected_features = [x_feature, y_feature]
            
            try:
                # Create statistics DataFrame with proper column names
                stats_dict = {}
                for feature in selected_features:
                    if feature in data.columns:  # Verify feature exists
                        stats = data.groupby('Cluster')[feature].agg(['mean', 'std', 'min', 'max']).round(2)
                        for stat in ['mean', 'std', 'min', 'max']:
                            stats_dict[f'{feature}_{stat}'] = stats[stat]
                
                if stats_dict:  # Only create DataFrame if we have statistics
                    cluster_stats = pd.DataFrame(stats_dict)
                    cluster_stats.index.name = 'Cluster'
                    st.dataframe(cluster_stats)
                else:
                    st.warning("Unable to generate statistics for the selected features.")
            
            except Exception as e:
                st.warning(f"Error generating statistics: {str(e)}")
            
            # Add feature distribution comparison
            st.write("### Feature Distribution Comparison")
            for feature in selected_features:
                try:
                    if feature in data.columns:  # Verify feature exists
                        fig = px.box(
                            data,
                            x='Cluster',
                            y=feature,
                            color='Cluster',
                            title=f'Distribution of {feature} by Cluster'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Error creating distribution plot for {feature}: {str(e)}")
            
            # Visualize clusters if 2D data
            if X.shape[1] == 2:
                st.subheader("2D Cluster Visualization")
                try:
                    fig3 = px.scatter(
                        data,
                        x=X.columns[0],
                        y=X.columns[1],
                        color='Cluster',
                        title='Data Points with Cluster Assignments'
                    )
                    st.plotly_chart(fig3)
                except Exception as e:
                    st.warning(f"Error creating 2D visualization: {str(e)}")
            
            # Display feature importance
            st.subheader("Feature Importance")
            st.write("""
            The feature importance plot shows how much each feature contributes to the cluster assignments.
            Features with higher importance are more influential in determining which cluster a customer belongs to.
            This helps identify the key characteristics that differentiate between customer segments.
            """)
        
        except Exception as e:
            st.error(f"An error occurred in the visualization section: {str(e)}")
            st.info("Please check your data and try again.")

    # Add data export and Power BI connection section
    st.subheader("Export and Power BI Integration")
    
    # Create a section for data export
    st.write("### Download Clustered Dataset")
    
    # Add cluster information to the original data
    if 'Cluster' in data.columns:
        # Create a copy of the data with cluster information
        export_data = data.copy()
        
        # Add cluster statistics as new columns
        for feature in X.columns:
            cluster_means = export_data.groupby('Cluster')[feature].transform('mean')
            cluster_stds = export_data.groupby('Cluster')[feature].transform('std')
            export_data[f'{feature}_cluster_mean'] = cluster_means
            export_data[f'{feature}_cluster_std'] = cluster_stds
        
        # Add cluster size information
        cluster_sizes = export_data['Cluster'].value_counts()
        export_data['cluster_size'] = export_data['Cluster'].map(cluster_sizes)
        export_data['cluster_percentage'] = export_data['Cluster'].map(cluster_sizes) / len(export_data) * 100
        
        # Create download button for CSV
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="Download Clustered Dataset as CSV",
            data=csv,
            file_name="clustered_customer_data.csv",
            mime="text/csv"
        )
        
        # Create download button for Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            export_data.to_excel(writer, sheet_name='Clustered Data', index=False)
            
            # Add cluster summary sheet
            cluster_summary = pd.DataFrame({
                'Cluster': range(n_clusters),
                'Size': [len(export_data[export_data['Cluster'] == i]) for i in range(n_clusters)],
                'Percentage': [len(export_data[export_data['Cluster'] == i]) / len(export_data) * 100 for i in range(n_clusters)]
            })
            cluster_summary.to_excel(writer, sheet_name='Cluster Summary', index=False)
            
            # Add feature statistics sheet
            feature_stats = export_data.groupby('Cluster')[X.columns].agg(['mean', 'std', 'min', 'max']).round(2)
            feature_stats.to_excel(writer, sheet_name='Feature Statistics')
        
        buffer.seek(0)
        st.download_button(
            label="Download Clustered Dataset as Excel",
            data=buffer,
            file_name="clustered_customer_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # Power BI Connection Instructions
        st.write("### Power BI Connection Instructions")
        st.write("""
        To connect this data to Power BI, you can use any of the following methods:
        
        1. **Direct CSV/Excel Import**:
           - Download the clustered dataset using the buttons above
           - In Power BI, go to 'Get Data' > 'Text/CSV' or 'Excel'
           - Select the downloaded file
           - The data will be imported with all cluster information
        
        2. **Power BI Service Connection**:
           - The dataset includes the following key columns for visualization:
             - Original features
             - Cluster assignments
             - Cluster statistics (means, standard deviations)
             - Cluster sizes and percentages
        
        3. **Recommended Power BI Visualizations**:
           - Scatter plots using original features, colored by cluster
           - Bar charts showing cluster sizes and distributions
           - Box plots for feature distributions by cluster
           - Tables showing cluster statistics
        """)
        
        # Add cluster comparison and best cluster analysis
        st.write("### Cluster Performance Analysis")
        
        # Add cluster-genre analysis
        st.subheader("Cluster-Genre Analysis")
        
        # Check if 'Genre' column exists in the data
        if 'Genre' in data.columns:
            # Create cluster-genre count DataFrame
            df2 = pd.DataFrame(data.groupby(['Cluster', 'Genre'])['Genre'].count())
            df2.columns = ['Count']
            df2 = df2.reset_index()
            
            # Display the raw data
            st.write("#### Cluster-Genre Distribution")
            st.dataframe(df2)
            
            # Create bar graph using plotly
            fig_genre = px.bar(
                df2,
                x='Cluster',
                y='Count',
                color='Genre',
                title='Distribution of Genres across Clusters',
                barmode='group',
                labels={'Cluster': 'Cluster Number', 'Count': 'Number of Customers'}
            )
            
            # Update layout
            fig_genre.update_layout(
                xaxis_title="Cluster",
                yaxis_title="Number of Customers",
                legend_title="Genre",
                showlegend=True
            )
            
            # Display the plot
            st.plotly_chart(fig_genre, use_container_width=True)
            
            # Add interpretation
            st.write("""
            ### Interpretation of Genre Distribution
            
            This visualization shows how different genres are distributed across the clusters:
            
            1. **Cluster Composition**: Shows the gender breakdown within each cluster
            2. **Gender Patterns**: Helps identify if certain clusters are more gender-specific
            3. **Targeting Insights**: Useful for gender-specific marketing strategies
            
            The bar graph allows you to:
            - Compare gender distribution across clusters
            - Identify gender-specific patterns
            - Understand the demographic composition of each cluster
            """)
        else:
            st.warning("Genre information not available in the dataset. Skipping genre analysis.")
        
        # Calculate cluster performance metrics
        cluster_metrics = {}
        for cluster in range(n_clusters):
            cluster_data = export_data[export_data['Cluster'] == cluster]
            
            # Calculate average values for each feature
            feature_means = cluster_data[X.columns].mean()
            
            # Calculate cluster metrics
            metrics = {
                'Size': len(cluster_data),
                'Percentage': len(cluster_data) / len(export_data) * 100,
                'Feature_Means': feature_means,
                'Score': 0  # Will be used for ranking
            }
            
            # Calculate a performance score based on feature values
            # This is a simple example - you can modify the scoring logic based on your business needs
            score = 0
            for feature in X.columns:
                # Normalize the feature value relative to all clusters
                normalized_value = (feature_means[feature] - export_data[feature].min()) / (export_data[feature].max() - export_data[feature].min())
                score += normalized_value
            
            metrics['Score'] = score / len(X.columns)  # Average score across features
            cluster_metrics[cluster] = metrics
        
        # Create a DataFrame for cluster comparison
        comparison_data = []
        for cluster, metrics in cluster_metrics.items():
            row = {
                'Cluster': cluster,
                'Size': metrics['Size'],
                'Percentage': metrics['Percentage'],
                'Performance Score': metrics['Score']
            }
            # Add feature means
            for feature, value in metrics['Feature_Means'].items():
                row[f'{feature} (Mean)'] = value
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display cluster comparison
        st.write("#### Cluster Comparison")
        st.dataframe(comparison_df.round(2))
        
        # Identify the best performing cluster
        best_cluster = max(cluster_metrics.items(), key=lambda x: x[1]['Score'])[0]
        best_metrics = cluster_metrics[best_cluster]
        
        st.write("#### Best Performing Cluster Analysis")
        st.write(f"**Cluster {best_cluster}** is identified as the best performing cluster based on the following characteristics:")
        
        # Display best cluster characteristics
        st.write(f"- **Size**: {best_metrics['Size']} customers ({best_metrics['Percentage']:.1f}% of total)")
        st.write("- **Key Features**:")
        for feature in X.columns:
            value = best_metrics['Feature_Means'][feature]
            overall_mean = export_data[feature].mean()
            diff_percentage = ((value - overall_mean) / overall_mean) * 100
            st.write(f"  - {feature}: {value:.2f} ({diff_percentage:+.1f}% compared to overall average)")
        
        st.write(f"- **Performance Score**: {best_metrics['Score']:.2f}")
        
        # Add strategic recommendations for the best cluster
        st.write("#### Strategic Recommendations for Best Cluster")
        st.write("""
        Based on the analysis of the best performing cluster, consider the following strategies:
        
        1. **Customer Retention**:
           - Identify common characteristics of customers in this cluster
           - Develop targeted retention programs
           - Monitor customer behavior patterns
        
        2. **Growth Opportunities**:
           - Use insights to identify potential customers with similar profiles
           - Develop targeted acquisition strategies
           - Create cluster-specific marketing campaigns
        
        3. **Product Development**:
           - Analyze feature preferences of this cluster
           - Develop products/services tailored to their needs
           - Consider premium offerings based on their behavior
        """)
        
        # Add data dictionary
        st.write("### Data Dictionary")
        data_dict = pd.DataFrame({
            'Column': export_data.columns,
            'Description': [
                'Original feature from the dataset' if col in X.columns else
                'Cluster assignment (0 to n-1)' if col == 'Cluster' else
                'Mean value of the feature within the cluster' if col.endswith('_cluster_mean') else
                'Standard deviation of the feature within the cluster' if col.endswith('_cluster_std') else
                'Number of customers in the cluster' if col == 'cluster_size' else
                'Percentage of total customers in the cluster' if col == 'cluster_percentage' else
                'Additional information'
                for col in export_data.columns
            ]
        })
        st.dataframe(data_dict)
        
    else:
        st.warning("Please generate clusters first to enable data export and Power BI integration.")
except Exception as e:
    st.error(f"An error occurred during data processing: {str(e)}")
    st.info("Please check your data and try again.")