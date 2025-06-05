import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import io
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Import your existing classes
try:
    from utils.logger import Logger
    from utils.data_processor import DataProcessor
    # For DomainDetector, we'll use a simplified version since the full one requires domain configs
    HAVE_UTILS = True
except ImportError:
    HAVE_UTILS = False
    st.warning("⚠️ Utils modules not found. Using simplified implementations.")

# Fallback implementations if utils are not available
if not HAVE_UTILS:
    class Logger:
        """Fallback logger class"""
        def __init__(self):
            self.logger = logging.getLogger(__name__)
            self.logs = []
        
        def log_info(self, message: str):
            self.logger.info(message)
            
        def log_error(self, message: str):
            self.logger.error(message)
            
        def log_warning(self, message: str):
            self.logger.warning(message)
    
    class DataProcessor:
        """Fallback data processor class"""
        def __init__(self, logger: Logger):
            self.logger = logger
            
        def process_domain_data(self, data: pd.DataFrame, domain: str) -> pd.DataFrame:
            """Basic data processing"""
            processed_data = data.copy()
            
            # Remove columns with all null values
            processed_data = processed_data.dropna(axis=1, how='all')
            
            # Fill numeric nulls with median
            numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                processed_data[col].fillna(processed_data[col].median(), inplace=True)
            
            # Fill categorical nulls with mode
            categorical_columns = processed_data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                processed_data[col].fillna(processed_data[col].mode()[0] if not processed_data[col].mode().empty else 'Unknown', inplace=True)
            
            self.logger.log_info(f"Data processed for domain: {domain}")
            return processed_data

class SimpleDomainDetector:
    """Simplified domain detector for the Streamlit app"""
    
    def __init__(self, logger: Logger):
        self.logger = logger
        
    def detect_domain(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect the domain of the dataset based on column names and content"""
        domain = None
        confidence = 0.0
        features = []
        
        columns = [col.lower() for col in data.columns]
        
        # Define domain keywords
        domain_keywords = {
            'customer_support': ['ticket', 'issue', 'resolution', 'customer', 'support', 'complaint', 'feedback'],
            'entertainment': ['movie', 'film', 'actor', 'director', 'rating', 'genre', 'music', 'artist'],
            'gaming': ['player', 'score', 'level', 'game', 'achievement', 'character', 'weapon'],
            'legal': ['case', 'court', 'lawyer', 'judge', 'law', 'legal', 'contract', 'clause'],
            'marketing': ['campaign', 'conversion', 'click', 'impression', 'ad', 'marketing', 'lead'],
            'logistics': ['shipment', 'delivery', 'warehouse', 'inventory', 'transport', 'freight'],
            'manufacturing': ['production', 'quality', 'defect', 'machine', 'factory', 'manufacturing'],
            'real_estate': ['bedroom', 'bathroom', 'sqft', 'area', 'location', 'property', 'house'],
            'agriculture': ['crop', 'yield', 'farm', 'harvest', 'soil', 'agriculture', 'livestock'],
            'energy': ['power', 'energy', 'consumption', 'electricity', 'solar', 'wind', 'fuel'],
            'hospitality': ['hotel', 'guest', 'room', 'booking', 'reservation', 'hospitality'],
            'automobile': ['car', 'vehicle', 'engine', 'mileage', 'fuel', 'automobile', 'model'],
            'telecommunications': ['call', 'data', 'network', 'signal', 'phone', 'telecom'],
            'government': ['citizen', 'service', 'department', 'government', 'public', 'policy'],
            'food_beverage': ['food', 'drink', 'restaurant', 'menu', 'cuisine', 'beverage'],
            'it_services': ['software', 'hardware', 'server', 'database', 'application', 'IT'],
            'event_management': ['event', 'venue', 'attendee', 'booking', 'schedule', 'conference'],
            'insurance': ['policy', 'claim', 'premium', 'coverage', 'insurance', 'beneficiary'],
            'retail': ['product', 'store', 'sales', 'customer', 'inventory', 'retail'],
            'hr_resources': ['employee', 'salary', 'department', 'performance', 'HR', 'hiring'],
            'finance': ['price', 'amount', 'balance', 'income', 'expense', 'profit', 'revenue'],
            'healthcare': ['age', 'weight', 'height', 'blood', 'pressure', 'heart', 'patient']
        }
        
        # Calculate scores for each domain
        scores = {}
        for domain_name, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if any(keyword in col for col in columns))
            if score > 0:
                scores[domain_name] = score
        
        if scores:
            domain = max(scores, key=scores.get)
            confidence = scores[domain] / len(columns)
            features = [col for col in data.columns if any(keyword in col.lower() 
                       for keyword in domain_keywords.get(domain, []))]
        
        return {
            'domain': domain,
            'confidence': confidence,
            'features': features,
            'config': None  # Simplified for this version
        }

class MultiDomainMLSystem:
    """Main application class"""
    
    def __init__(self):
        self.logger = Logger()
        # Use SimpleDomainDetector instead of the complex one that requires domain configs
        self.detector = SimpleDomainDetector(logger=self.logger)
        self.processor = DataProcessor(logger=self.logger)
        
        # Initialize session state variables
        if 'dataset_uploaded' not in st.session_state:
            st.session_state['dataset_uploaded'] = False
        if 'detected_domain' not in st.session_state:
            st.session_state['detected_domain'] = None
        if 'processed_data' not in st.session_state:
            st.session_state['processed_data'] = None
        if 'raw_data' not in st.session_state:
            st.session_state['raw_data'] = None
        if 'models' not in st.session_state:
            st.session_state['models'] = {}

    def run(self):
        st.set_page_config(page_title="Multi-Domain ML System", layout="wide")
        st.title("🧠 Multi-Domain ML System")
        st.markdown("### Automatically detects and analyzes datasets from multiple domains")

        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Choose a section", [
            "Dataset Upload & Detection", 
            "Data Analysis",
            "Model Training", 
            "Intelligent Query",
            "Predictions", 
            "System Logs"
        ])

        if page == "Dataset Upload & Detection":
            self.upload_and_detect_page()
        elif page == "Data Analysis":
            self.data_analysis_page()
        elif page == "Model Training":
            self.model_training_page()
        elif page == "Intelligent Query":
            self.intelligent_query_page()
        elif page == "Predictions":
            self.predictions_page()
        elif page == "System Logs":
            self.system_logs_page()

    def upload_and_detect_page(self):
        st.subheader("📤 Upload Dataset")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                self.logger.log_info(f"Dataset loaded successfully! Shape: {data.shape}")
                st.success(f"✅ Dataset loaded! Shape: {data.shape}")

                # Auto-detect domain
                detected = self.detector.detect_domain(data)
                detected_domain = detected['domain']
                confidence = detected['confidence']

                if detected_domain:
                    st.info(f"🎯 Detected domain: **{detected_domain.replace('_', ' ').title()}** (confidence: {confidence:.2f})")
                    st.write(f"**Key features identified**: {', '.join(detected['features'][:5])}")
                else:
                    st.warning("⚠️ Could not confidently detect a specific domain. Using generic analysis.")

                # Save in session state
                st.session_state['raw_data'] = data
                st.session_state['detected_domain'] = detected_domain
                st.session_state['dataset_uploaded'] = True
                
                # Process data
                processed_data = self.processor.process_domain_data(data, detected_domain) if detected_domain else data
                st.session_state['processed_data'] = processed_data

                # Display basic info
                st.subheader("📊 Dataset Overview")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Rows", data.shape[0])
                with col2:
                    st.metric("Columns", data.shape[1])
                with col3:
                    st.metric("Numeric Columns", len(data.select_dtypes(include=[np.number]).columns))
                with col4:
                    st.metric("Missing Values", data.isnull().sum().sum())

                # Show sample data
                st.subheader("Sample Data")
                st.dataframe(processed_data.head(10))

            except Exception as e:
                self.logger.log_error(f"Dataset upload failed: {str(e)}")
                st.error(f"❌ Error loading dataset: {e}")

    def data_analysis_page(self):
        st.header("📈 Data Analysis")
        
        if not st.session_state.get('dataset_uploaded', False):
            st.warning("⚠️ Please upload a dataset first in the 'Dataset Upload & Detection' section.")
            return

        data = st.session_state['processed_data']
        detected_domain = st.session_state.get('detected_domain')
        
        if detected_domain:
            st.info(f"📊 Analysis for **{detected_domain.replace('_', ' ').title()}** domain")

        # Basic statistics
        st.subheader("📋 Dataset Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types**")
            dtype_df = pd.DataFrame(data.dtypes).reset_index()
            dtype_df.columns = ['Column', 'Type']
            st.dataframe(dtype_df)
        
        with col2:
            st.write("**Missing Values**")
            missing_df = pd.DataFrame(data.isnull().sum()).reset_index()
            missing_df.columns = ['Column', 'Missing Count']
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if not missing_df.empty:
                st.dataframe(missing_df)
            else:
                st.write("No missing values found! ✅")

        # Statistical summary
        st.subheader("📊 Statistical Summary")
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            st.dataframe(numeric_data.describe())
        else:
            st.warning("No numeric columns found for statistical analysis.")

        # Visualizations
        st.subheader("📈 Visualizations")
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X-axis", numeric_cols, key="x_axis")
            with col2:
                y_col = st.selectbox("Select Y-axis", numeric_cols, key="y_axis")
            
            if x_col != y_col:
                fig = px.scatter(data, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plots
        if numeric_cols:
            st.subheader("📊 Distribution Analysis")
            selected_col = st.selectbox("Select column for distribution", numeric_cols, key="dist_col")
            fig = px.histogram(data, x=selected_col, title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)

    def model_training_page(self):
        st.header("🛠️ Model Training")
        
        if not st.session_state.get('dataset_uploaded', False):
            st.warning("⚠️ Please upload a dataset first.")
            return

        data = st.session_state['processed_data']
        
        st.subheader("🎯 Configure Model Training")
        
        # Select target column
        target_col = st.selectbox("Select Target Column", data.columns.tolist(), key="target_col")
        
        if target_col:
            # Determine if it's regression or classification
            if data[target_col].dtype in ['int64', 'float64']:
                # Could be either - let user choose
                model_type = st.selectbox("Model Type", ["Regression", "Classification"], key="model_type")
            else:
                model_type = "Classification"
                st.info("Target column appears to be categorical. Using Classification.")
            
            # Select feature columns
            available_features = [col for col in data.columns if col != target_col and data[col].dtype in ['int64', 'float64']]
            
            if not available_features:
                st.error("❌ No numeric features available for training.")
                return
            
            selected_features = st.multiselect("Select Feature Columns", available_features, default=available_features[:5])
            
            if st.button("🚀 Train Model", type="primary"):
                if not selected_features:
                    st.error("Please select at least one feature column.")
                    return
                
                try:
                    with st.spinner("Training model..."):
                        X = data[selected_features].dropna()
                        y = data.loc[X.index, target_col]
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        if model_type == "Regression":
                            model = RandomForestRegressor(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            score = r2_score(y_test, y_pred)
                            
                            st.session_state['models'][target_col] = {
                                'model': model, 
                                'model_type': 'regression', 
                                'r2': score,
                                'features': selected_features
                            }
                            
                            st.success(f"✅ Regression model trained! R² Score: {score:.3f}")
                            
                        else:  # Classification
                            model = RandomForestClassifier(n_estimators=100, random_state=42)
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            score = accuracy_score(y_test, y_pred)
                            
                            st.session_state['models'][target_col] = {
                                'model': model, 
                                'model_type': 'classification', 
                                'accuracy': score,
                                'features': selected_features
                            }
                            
                            st.success(f"✅ Classification model trained! Accuracy: {score:.3f}")
                        
                        self.logger.log_info(f"Model trained for {target_col}")
                        
                        # Feature importance
                        importance = pd.DataFrame({
                            'feature': selected_features,
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        st.subheader("📊 Feature Importance")
                        fig = px.bar(importance, x='importance', y='feature', orientation='h',
                                   title="Feature Importance")
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
                    self.logger.log_error(f"Model training error: {str(e)}")

    def predictions_page(self):
        st.header("📉 Predictions")
        
        if not st.session_state.get('models'):
            st.warning("⚠️ No trained models available. Please train a model first in the 'Model Training' section.")
            return

        data = st.session_state['processed_data']
        models = st.session_state['models']
        
        st.subheader("🔮 Generate Predictions")
        
        # Select model
        target_col = st.selectbox("Select Model (Target Column)", list(models.keys()), key="pred_target")
        
        if st.button("🎯 Generate Predictions", type="primary"):
            try:
                with st.spinner("Generating predictions..."):
                    model_info = models[target_col]
                    features = model_info['features']
                    
                    X = data[features].dropna()
                    predictions = model_info['model'].predict(X)
                    
                    # Create results dataframe
                    results_df = data.loc[X.index].copy()
                    results_df['Predictions'] = predictions
                    
                    st.success(f"✅ Generated {len(predictions)} predictions!")
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Sample Predictions**")
                        display_cols = [target_col, 'Predictions'] + features[:3]
                        st.dataframe(results_df[display_cols].head(10))
                    
                    with col2:
                        st.write("**Prediction Statistics**")
                        pred_stats = pd.DataFrame({
                            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                            'Value': [
                                np.mean(predictions),
                                np.median(predictions),
                                np.std(predictions),
                                np.min(predictions),
                                np.max(predictions)
                            ]
                        })
                        st.dataframe(pred_stats)
                    
                    # Visualization
                    st.subheader("📈 Prediction Distribution")
                    fig = px.histogram(x=predictions, title=f"Distribution of Predictions for {target_col}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Actual vs Predicted (if target column exists)
                    if target_col in results_df.columns:
                        fig2 = px.scatter(x=results_df[target_col], y=predictions, 
                                        title=f"Actual vs Predicted: {target_col}",
                                        labels={'x': f'Actual {target_col}', 'y': f'Predicted {target_col}'})
                        st.plotly_chart(fig2, use_container_width=True)
                    
                    self.logger.log_info(f"Predictions generated for {target_col}")
                    
            except Exception as e:
                st.error(f"❌ Error generating predictions: {str(e)}")
                self.logger.log_error(f"Prediction error: {str(e)}")

    def intelligent_query_page(self):
        st.header("💡 Intelligent Query")

        if not st.session_state.get('dataset_uploaded', False):
            st.warning("⚠️ Please upload a dataset first in the 'Dataset Upload & Detection' section.")
            return

        data = st.session_state['processed_data']
        
        st.markdown("🔍 **Ask questions about your dataset!**")
        st.markdown("Try asking: *'Show top 10 rows'*, *'What is the average price?'*, *'Describe the data'*")

        question = st.text_input("💭 Enter your question:", placeholder="What would you like to know about your data?")

        if question:
            question_lower = question.lower()
            
            try:
                # Show rows
                if any(word in question_lower for word in ["head", "top", "first", "show"]):
                    num = 5  # default
                    for word in question_lower.split():
                        if word.isdigit():
                            num = min(int(word), 50)  # limit to 50 rows
                            break
                    
                    st.write(f"📋 **Top {num} rows:**")
                    st.dataframe(data.head(num))

                # Column information
                elif any(word in question_lower for word in ["columns", "features", "variables"]):
                    st.write("📊 **Available columns:**")
                    col_info = pd.DataFrame({
                        'Column': data.columns,
                        'Type': data.dtypes,
                        'Non-Null Count': data.count()
                    })
                    st.dataframe(col_info)

                # Dataset size/shape
                elif any(word in question_lower for word in ["shape", "size", "dimensions"]):
                    st.write(f"📏 **Dataset dimensions:** {data.shape[0]} rows × {data.shape[1]} columns")

                # Statistical summary
                elif any(word in question_lower for word in ["describe", "summary", "statistics", "stats"]):
                    st.write("📈 **Statistical Summary:**")
                    numeric_data = data.select_dtypes(include=[np.number])
                    if not numeric_data.empty:
                        st.dataframe(numeric_data.describe())
                    else:
                        st.write("No numeric columns available for statistical summary.")

                # Missing values
                elif any(word in question_lower for word in ["null", "missing", "na", "empty"]):
                    st.write("🔍 **Missing values per column:**")
                    missing_info = data.isnull().sum()
                    missing_df = missing_info[missing_info > 0].to_frame(name='Missing Count')
                    if not missing_df.empty:
                        st.dataframe(missing_df)
                    else:
                        st.write("✅ No missing values found!")

                # Average/Mean calculations
                elif any(word in question_lower for word in ["average", "mean"]):
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    found_col = None
                    
                    for col in numeric_cols:
                        if col.lower() in question_lower:
                            found_col = col
                            break
                    
                    if found_col:
                        avg_val = data[found_col].mean()
                        st.write(f"📊 **Average {found_col}:** {avg_val:.2f}")
                    else:
                        st.write("📊 **Averages for all numeric columns:**")
                        if not numeric_cols.empty:
                            avg_df = data[numeric_cols].mean().to_frame(name='Average')
                            st.dataframe(avg_df)
                        else:
                            st.write("No numeric columns found.")

                # Maximum values
                elif "max" in question_lower:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    found_col = None
                    
                    for col in numeric_cols:
                        if col.lower() in question_lower:
                            found_col = col
                            break
                    
                    if found_col:
                        max_val = data[found_col].max()
                        st.write(f"📈 **Maximum {found_col}:** {max_val}")
                    else:
                        st.write("📈 **Maximum values for all numeric columns:**")
                        if not numeric_cols.empty:
                            max_df = data[numeric_cols].max().to_frame(name='Maximum')
                            st.dataframe(max_df)

                # Minimum values
                elif "min" in question_lower:
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    found_col = None
                    
                    for col in numeric_cols:
                        if col.lower() in question_lower:
                            found_col = col
                            break
                    
                    if found_col:
                        min_val = data[found_col].min()
                        st.write(f"📉 **Minimum {found_col}:** {min_val}")
                    else:
                        st.write("📉 **Minimum values for all numeric columns:**")
                        if not numeric_cols.empty:
                            min_df = data[numeric_cols].min().to_frame(name='Minimum')
                            st.dataframe(min_df)

                # Data types
                elif any(word in question_lower for word in ["types", "dtype", "datatype"]):
                    st.write("🏷️ **Data types:**")
                    dtype_df = data.dtypes.to_frame(name='Data Type')
                    st.dataframe(dtype_df)

                # Unique values
                elif any(word in question_lower for word in ["unique", "distinct"]):
                    st.write("🔢 **Unique values per column:**")
                    unique_df = data.nunique().to_frame(name='Unique Count')
                    st.dataframe(unique_df)

                else:
                    st.warning("❓ I couldn't understand your question. Try asking about:")
                    st.markdown("""
                    - **Data overview:** 'show top rows', 'describe data', 'what columns'
                    - **Statistics:** 'average price', 'maximum value', 'minimum age'
                    - **Data quality:** 'missing values', 'null counts'
                    - **Data info:** 'data types', 'unique values', 'dataset size'
                    """)

            except Exception as e:
                st.error(f"❌ Error processing your question: {str(e)}")
                self.logger.log_error(f"Query processing error: {str(e)}")

    def system_logs_page(self):
        st.header("🧾 System Logs")
        
        st.info("📝 System logs and activity monitoring")
        
        # Session information
        st.subheader("📊 Session Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Dataset Uploaded", "Yes" if st.session_state.get('dataset_uploaded') else "No")
        with col2:
            st.metric("Models Trained", len(st.session_state.get('models', {})))
        with col3:
            detected_domain = st.session_state.get('detected_domain', 'None')
            st.metric("Detected Domain", detected_domain.replace('_', ' ').title() if detected_domain else 'None')
        
        # Session state details
        if st.checkbox("Show detailed session state"):
            st.subheader("🔍 Session State Details")
            session_info = {
                'Dataset Uploaded': st.session_state.get('dataset_uploaded', False),
                'Detected Domain': st.session_state.get('detected_domain'),
                'Data Shape': st.session_state.get('processed_data').shape if st.session_state.get('processed_data') is not None else None,
                'Available Models': list(st.session_state.get('models', {}).keys()),
                'Session Keys': list(st.session_state.keys())
            }
            
            for key, value in session_info.items():
                st.write(f"**{key}:** {value}")


if __name__ == "__main__":
    app = MultiDomainMLSystem()
    app.run()