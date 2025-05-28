import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, _tree
from sklearn.datasets import make_blobs
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Segmentation Platform", layout="wide")

# ---------------- Database Functions ----------------
def create_usertable():
    """Create users table if it doesn't exist"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT PRIMARY KEY, password TEXT, created_date TEXT)')
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password for security"""
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password):
    """Add new user to database"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    try:
        c.execute('INSERT INTO users(username, password, created_date) VALUES (?, ?, ?)', 
                 (username, hashed_password, pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def login_user(username, password):
    """Verify user login credentials"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_password))
    data = c.fetchone()
    conn.close()
    return data

def check_username_exists(username):
    """Check if username already exists"""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE username = ?', (username,))
    data = c.fetchone()
    conn.close()
    return data is not None

# ---------------- Main Customer Segmentation App ----------------
def main_segmentation_app(username):
    """Main application with K-means clustering functionality"""
    
    # Header with user info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🎯 Customer Segmentation Platform")
        st.write("Advanced K-means clustering and customer insights dashboard")
    
    with col2:
        st.info(f"👤 Logged in as: **{username}**")
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = ""
            st.rerun()
    
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("📊 Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Data Upload & Analysis", "Synthetic Data Testing", "User Guide"]
    )
    
    if app_mode == "User Guide":
        show_user_guide()
        return
    
    # Main application logic
    st.write("""
    This application performs advanced K-means clustering analysis to help you understand customer segments 
    and provides interpretable insights through interactive visualizations and AI-powered explanations.
    """)

    # Data source selection
    if app_mode == "Synthetic Data Testing":
        use_synthetic = True
        st.info("🧪 **Synthetic Data Mode** - Perfect for testing and learning!")
    else:
        use_synthetic = st.checkbox("🧪 Use Synthetic Data for Testing", value=False, key="use_synthetic")

    if use_synthetic:
        data, X = handle_synthetic_data()
    else:
        data, X = handle_file_upload()
    
    if data is not None and X is not None:
        perform_clustering_analysis(data, X, username)

def show_user_guide():
    """Display user guide and instructions"""
    st.header("📚 User Guide")
    
    st.subheader("🚀 Getting Started")
    st.write("""
    1. **Choose Your Data Source:**
       - Upload your own CSV/Excel file with customer data
       - Or use synthetic data for testing and learning
    
    2. **Data Requirements:**
       - At least 2 numeric columns for clustering
       - Customer data (e.g., age, income, spending score, etc.)
       - Missing values will be handled automatically
    
    3. **Analysis Process:**
       - Select features for clustering
       - Use the Elbow Method to find optimal cluster count
       - Generate clusters and explore insights
    """)
    
    st.subheader("📊 Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        **🔍 Analysis Tools:**
        - K-means clustering with k-means++ initialization
        - Elbow Method for optimal cluster selection
        - Interactive visualizations
        - AI-powered cluster explanations
        - Feature importance analysis
        """)
    
    with col2:
        st.write("""
        **📈 Export Options:**
        - Download clustered datasets (CSV/Excel)
        - Power BI integration instructions
        - Cluster statistics and summaries
        - Business insights and recommendations
        """)
    
    st.subheader("💡 Best Practices")
    st.write("""
    - Start with 2-5 clusters for initial analysis
    - Use the Elbow Method to guide cluster selection
    - Review feature importance to understand key differentiators
    - Consider business context when interpreting results
    - Export results for further analysis in other tools
    """)

def handle_synthetic_data():
    """Handle synthetic data generation"""
    st.subheader("🧪 Synthetic Data Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        n_samples = st.slider("Number of samples", 100, 1000, 300, key="synthetic_samples")
    with col2:
        n_features = st.slider("Number of features", 2, 5, 3, key="synthetic_features")
    with col3:
        n_clusters = st.slider("True number of clusters", 2, 5, 3, key="synthetic_clusters")
    
    # Generate synthetic data
    X, true_labels = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=2.5,
        random_state=42
    )
    
    # Convert to DataFrame with meaningful names
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    data = pd.DataFrame(X, columns=feature_names)
    data['True_Cluster'] = true_labels
    
    # Display data preview
    st.subheader("📊 Synthetic Data Preview")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(data.head(10))
    
    with col2:
        st.metric("Total Samples", len(data))
        st.metric("Features", n_features)
        st.metric("True Clusters", n_clusters)
    
    # Visualize if 2D
    if n_features == 2:
        st.subheader("🎯 True Cluster Visualization")
        fig = px.scatter(
            data,
            x='Feature_1',
            y='Feature_2',
            color='True_Cluster',
            title='Synthetic Data with True Clusters',
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    X = data[feature_names].copy()
    return data, X

def handle_file_upload():
    """Handle file upload and processing"""
    uploaded_file = st.file_uploader(
        "📁 Upload your customer data (CSV or Excel file)", 
        type=['csv', 'xlsx', 'xls'], 
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Read the data
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            # Display data info
            st.subheader("📊 Data Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", len(data))
            with col2:
                st.metric("Total Columns", len(data.columns))
            with col3:
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                st.metric("Numeric Columns", len(numeric_cols))
            
            # Data preview
            st.write("**Data Preview:**")
            st.dataframe(data.head())
            
            # Check for missing values
            missing_values = data.isnull().sum()
            if missing_values.any():
                st.warning("⚠️ Missing values detected - they will be handled automatically")
                with st.expander("View Missing Values Details"):
                    missing_df = pd.DataFrame({
                        'Column': missing_values.index,
                        'Missing Count': missing_values.values,
                        'Missing %': (missing_values.values / len(data) * 100).round(2)
                    })
                    st.dataframe(missing_df[missing_df['Missing Count'] > 0])
            
            # Feature selection
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) < 2:
                st.error("❌ Dataset must contain at least 2 numeric columns for clustering.")
                return None, None
            
            st.subheader("🎯 Feature Selection")
            features = st.multiselect(
                "Select features for clustering (numeric columns only)",
                options=numeric_cols,
                default=numeric_cols[:min(4, len(numeric_cols))],
                key="feature_selector"
            )
            
            if len(features) >= 2:
                X = data[features].copy()
                return data, X
            else:
                st.warning("⚠️ Please select at least 2 features for clustering.")
                return None, None
                
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            return None, None
    else:
        st.info("👆 Please upload a CSV or Excel file to begin analysis")
        return None, None

def perform_clustering_analysis(data, X, username):
    """Main clustering analysis function"""
    
    # Handle missing values
    if X.isnull().any().any():
        st.info("🔧 Handling missing values using mean imputation...")
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
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Elbow Method Analysis
    st.subheader("📈 Elbow Method Analysis")
    st.write("The Elbow Method helps determine the optimal number of clusters by analyzing Within-Cluster Sum of Squares (WCSS).")
    
    max_clusters = min(10, len(data) - 1)
    wcss = []
    k_range = range(1, max_clusters + 1)
    
    # Calculate WCSS with progress bar
    progress_bar = st.progress(0)
    for i, k in enumerate(k_range):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
        progress_bar.progress((i + 1) / len(k_range))
    
    # Create Elbow plot
    fig_elbow = go.Figure()
    fig_elbow.add_trace(go.Scatter(
        x=list(k_range),
        y=wcss,
        mode='lines+markers',
        name='WCSS',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig_elbow.update_layout(
        title='📊 Elbow Method for Optimal K',
        xaxis_title='Number of Clusters (K)',
        yaxis_title='Within-Cluster Sum of Squares (WCSS)',
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig_elbow, use_container_width=True)
    
    # Cluster selection
    col1, col2 = st.columns([2, 1])
    with col1:
        n_clusters = st.slider("🎯 Select number of clusters", 2, max_clusters, 3, key="kmeans_n_clusters")
    with col2:
        st.metric("Selected Clusters", n_clusters)
    
    if st.button("🚀 Generate Cluster Analysis", key="generate_clusters", type="primary"):
        with st.spinner("🔄 Performing clustering analysis..."):
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Add clusters to data
            data_with_clusters = data.copy()
            data_with_clusters['Cluster'] = clusters
            
            # Display results
            display_clustering_results(data_with_clusters, X, X_scaled, kmeans, scaler, n_clusters, username)

def display_clustering_results(data, X, X_scaled, kmeans, scaler, n_clusters, username):
    """Display comprehensive clustering results"""
    
    st.success(f"✅ Successfully created {n_clusters} customer segments!")
    
    # Cluster Overview
    st.subheader("📊 Cluster Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Customers", len(data))
    with col2:
        st.metric("Number of Clusters", n_clusters)
    with col3:
        st.metric("Features Used", len(X.columns))
    with col4:
        st.metric("Algorithm Iterations", kmeans.n_iter_)
    
    # Cluster sizes
    cluster_sizes = data['Cluster'].value_counts().sort_index()
    
    # Main visualization
    st.subheader("🎯 Cluster Visualization")
    
    if len(X.columns) >= 2:
        colors = px.colors.qualitative.Set1[:n_clusters]
        
        fig = go.Figure()
        
        # Add cluster points
        for cluster in range(n_clusters):
            cluster_data = data[data['Cluster'] == cluster]
            fig.add_trace(go.Scatter(
                x=cluster_data[X.columns[0]],
                y=cluster_data[X.columns[1]],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(
                    size=8,
                    color=colors[cluster],
                    opacity=0.7
                ),
                hovertemplate=f"<b>Cluster {cluster}</b><br>" +
                             f"{X.columns[0]}: %{{x:.2f}}<br>" +
                             f"{X.columns[1]}: %{{y:.2f}}<br>" +
                             "<extra></extra>"
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
            name='Centroids',
            marker=dict(
                size=15,
                color='black',
                symbol='diamond',
                line=dict(width=2, color='white')
            ),
            hovertemplate="<b>Centroid</b><br>" +
                         f"{X.columns[0]}: %{{x:.2f}}<br>" +
                         f"{X.columns[1]}: %{{y:.2f}}<br>" +
                         "<extra></extra>"
        ))
        
        fig.update_layout(
            title=f'Customer Segments ({n_clusters} Clusters)',
            xaxis_title=X.columns[0],
            yaxis_title=X.columns[1],
            template='plotly_white',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Statistics
    st.subheader("📈 Cluster Statistics")
    
    # Summary table
    cluster_summary = []
    for cluster in range(n_clusters):
        cluster_data = data[data['Cluster'] == cluster]
        summary = {
            'Cluster': cluster,
            'Size': len(cluster_data),
            'Percentage': f"{len(cluster_data)/len(data)*100:.1f}%"
        }
        
        # Add feature means
        for feature in X.columns:
            summary[f'{feature} (Avg)'] = f"{cluster_data[feature].mean():.2f}"
        
        cluster_summary.append(summary)
    
    summary_df = pd.DataFrame(cluster_summary)
    st.dataframe(summary_df, use_container_width=True)
    
    # Detailed statistics
    with st.expander("📊 View Detailed Statistics"):
        detailed_stats = data.groupby('Cluster')[X.columns].agg(['mean', 'std', 'min', 'max']).round(2)
        st.dataframe(detailed_stats)
    
    # Feature Analysis
    st.subheader("🔍 Feature Analysis")
    
    # Feature comparison across clusters
    fig_features = make_subplots(
        rows=1, cols=len(X.columns),
        subplot_titles=[f'Average {col}' for col in X.columns]
    )
    
    cluster_means = data.groupby('Cluster')[X.columns].mean()
    
    for i, feature in enumerate(X.columns, 1):
        fig_features.add_trace(
            go.Bar(
                x=cluster_means.index,
                y=cluster_means[feature],
                name=feature,
                marker_color=colors[i-1] if i <= len(colors) else colors[0],
                showlegend=False
            ),
            row=1, col=i
        )
        
        fig_features.update_xaxes(title_text="Cluster", row=1, col=i)
        fig_features.update_yaxes(title_text="Average Value", row=1, col=i)
    
    fig_features.update_layout(
        title_text="Feature Comparison Across Clusters",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig_features, use_container_width=True)
    
    # AI-Powered Insights
    generate_ai_insights(data, X, n_clusters)
    
    # Export Options
    provide_export_options(data, X, username)

def generate_ai_insights(data, X, n_clusters):
    """Generate AI-powered cluster insights"""
    st.subheader("🤖 AI-Powered Cluster Insights")
    
    for cluster in range(n_clusters):
        cluster_data = data[data['Cluster'] == cluster]
        
        with st.expander(f"🎯 Cluster {cluster} Analysis ({len(cluster_data)} customers)"):
            
            # Generate insights
            insights = []
            
            # Size analysis
            percentage = len(cluster_data) / len(data) * 100
            if percentage > 30:
                insights.append(f"This is a **major segment** representing {percentage:.1f}% of your customer base.")
            elif percentage < 10:
                insights.append(f"This is a **niche segment** representing {percentage:.1f}% of your customer base.")
            else:
                insights.append(f"This is a **moderate segment** representing {percentage:.1f}% of your customer base.")
            
            # Feature analysis
            for feature in X.columns:
                cluster_mean = cluster_data[feature].mean()
                overall_mean = data[feature].mean()
                diff_percentage = ((cluster_mean - overall_mean) / overall_mean) * 100
                
                if abs(diff_percentage) > 15:
                    direction = "higher" if diff_percentage > 0 else "lower"
                    insights.append(f"**{feature}**: {abs(diff_percentage):.1f}% {direction} than average ({cluster_mean:.2f} vs {overall_mean:.2f})")
            
            # Display insights
            for insight in insights:
                st.write(f"• {insight}")
            
            # Feature distribution chart
            fig_dist = go.Figure()
            for feature in X.columns:
                fig_dist.add_trace(go.Box(
                    y=cluster_data[feature],
                    name=feature,
                    boxpoints='outliers'
                ))
            
            fig_dist.update_layout(
                title=f'Feature Distribution - Cluster {cluster}',
                yaxis_title='Values',
                template='plotly_white',
                height=300
            )
            
            st.plotly_chart(fig_dist, use_container_width=True)

def provide_export_options(data, X, username):
    """Provide data export options"""
    st.subheader("💾 Export & Integration")
    
    # Prepare export data
    export_data = data.copy()
    
    # Add cluster statistics
    for feature in X.columns:
        cluster_means = export_data.groupby('Cluster')[feature].transform('mean')
        export_data[f'{feature}_cluster_avg'] = cluster_means
    
    # Add cluster info
    cluster_sizes = export_data['Cluster'].value_counts()
    export_data['cluster_size'] = export_data['Cluster'].map(cluster_sizes)
    export_data['cluster_percentage'] = (export_data['Cluster'].map(cluster_sizes) / len(export_data) * 100).round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV Download
        csv = export_data.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name=f"customer_segments_{username}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_csv"
        )
    
    with col2:
        # Excel Download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            export_data.to_excel(writer, sheet_name='Customer_Segments', index=False)
            
            # Summary sheet
            summary = pd.DataFrame({
                'Cluster': range(len(cluster_sizes)),
                'Size': cluster_sizes.values,
                'Percentage': (cluster_sizes.values / len(export_data) * 100).round(2)
            })
            summary.to_excel(writer, sheet_name='Cluster_Summary', index=False)
        
        buffer.seek(0)
        st.download_button(
            label="📊 Download as Excel",
            data=buffer,
            file_name=f"customer_segments_{username}_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel"
        )
    
    # Power BI Integration Guide
    with st.expander("🔌 Power BI Integration Guide"):
        st.write("""
        **Connect to Power BI:**
        1. Download the Excel file above
        2. In Power BI Desktop: Get Data → Excel → Select your downloaded file
        3. Import both sheets: 'Customer_Segments' and 'Cluster_Summary'
        
        **Recommended Visualizations:**
        - Scatter plot: Features colored by Cluster
        - Pie chart: Cluster distribution
        - Bar chart: Average features by cluster
        - Table: Detailed customer segments
        """)

# ---------------- Authentication Functions ----------------
def login_signup_page():
    """Login and signup page"""
    st.title("🔐 Customer Segmentation Platform")
    st.write("Welcome to the advanced customer analytics dashboard")
    
    # Create tabs for login and signup
    tab1, tab2 = st.tabs(["🔑 Login", "📝 Sign Up"])
    
    create_usertable()
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", key="login_username")
            password = st.text_input("🔒 Password", type="password", key="login_password")
            
            login_button = st.form_submit_button("🚀 Login", type="primary")
            
            if login_button:
                if username and password:
                    result = login_user(username, password)
                    if result:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = username
                        st.success(f"Welcome back, {username}! 🎉")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("signup_form"):
            new_username = st.text_input("👤 Choose Username", key="signup_username")
            new_password = st.text_input("🔒 Choose Password", type="password", key="signup_password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", key="confirm_password")
            
            signup_button = st.form_submit_button("✨ Create Account", type="primary")
            
            if signup_button:
                if new_username and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("❌ Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters long")
                    elif check_username_exists(new_username):
                        st.error("❌ Username already exists. Please choose a different one.")
                    else:
                        success = add_user(new_username, new_password)
                        if success:
                            st.success("✅ Account created successfully! Please login with your credentials.")
                        else:
                            st.error("❌ Error creating account. Please try again.")
                else:
                    st.warning("⚠️ Please fill in all fields")

# ---------------- Main Application Entry Point ----------------
def main():
    """Main application entry point"""
    
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
    
    # Route to appropriate page
    if st.session_state["logged_in"]:
        main_segmentation_app(st.session_state["username"])
    else:
        login_signup_page()

# Run the application
if __name__ == "__main__":
    main()
