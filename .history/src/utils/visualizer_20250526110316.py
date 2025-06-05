import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, Any

class Visualizer:
    def __init__(self):
        pass
    
    def create_education_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any],
                                detected_features: Dict[str, Any]):
        """Create comprehensive education data analysis"""
        st.subheader("📊 Education Data Analysis")
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", len(raw_data))
        with col2:
            grade_cols = [col for col in raw_data.columns 
                         if any(keyword in col.lower() for keyword in ['grade', 'score'])]
            if grade_cols:
                avg_grade = raw_data[grade_cols[0]].mean()
                st.metric("Average Grade", f"{avg_grade:.1f}")
            else:
                st.metric("Numeric Features", len(raw_data.select_dtypes(include=[np.number]).columns))
        with col3:
            if grade_cols:
                std_grade = raw_data[grade_cols[0]].std()
                st.metric("Grade Std Dev", f"{std_grade:.1f}")
            else:
                st.metric("Categorical Features", len(raw_data.select_dtypes(include=['object']).columns))
        with col4:
            st.metric("Features", len(raw_data.columns))
        
        # Visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Demographics", "Correlations", "Distributions"])
        
        with tab1:
            self._create_performance_analysis(raw_data)
        
        with tab2:
            self._create_demographic_analysis(raw_data)
        
        with tab3:
            self._create_correlation_analysis(raw_data)
        
        with tab4:
            self._create_distribution_analysis(raw_data)
    
    def create_basic_analysis(self, data: pd.DataFrame):
        """Create basic analysis for non-education datasets"""
        st.subheader("📊 Basic Data Analysis")
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", data.shape[0])
        with col2:
            st.metric("Columns", data.shape[1])
        with col3:
            st.metric("Missing Values", data.isnull().sum().sum())
        
        # Data types
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': data.columns,
            'Data Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Null Count': data.isnull().sum()
        })
        st.dataframe(dtype_df)
        
        # Basic visualizations
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.subheader("Numeric Data Distribution")
            col = st.selectbox("Select column to visualize", numeric_cols)
            fig = px.histogram(data, x=col, title=f'Distribution of {col}')
            st.plotly_chart(fig, use_container_width=True)
    
    def _create_performance_analysis(self, data: pd.DataFrame):
        """Create performance-specific visualizations"""
        grade_cols = [col for col in data.columns 
                     if any(keyword in col.lower() for keyword in ['grade', 'score', 'result', 'gpa'])]
        
        if not grade_cols:
            st.info("No grade/score columns detected for performance analysis.")
            return
        
        target_col = grade_cols[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Grade distribution
            fig1 = px.histogram(data, x=target_col, nbins=20, 
                               title=f'Distribution of {target_col}')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Performance categories
            if data[target_col].max() <= 100:
                # Create grade categories
                grade_categories = pd.cut(data[target_col], 
                                        bins=[0, 60, 70, 80, 90, 100],
                                        labels=['F', 'D', 'C', 'B', 'A'])
                category_counts = grade_categories.value_counts()
                
                fig2 = px.pie(values=category_counts.values, names=category_counts.index,
                             title='Grade Distribution by Category')
                st.plotly_chart(fig2, use_container_width=True)
            else:
                # Box plot for continuous scores
                fig2 = px.box(y=data[target_col], title=f'Box Plot of {target_col}')
                st.plotly_chart(fig2, use_container_width=True)
        
        # Performance vs other factors
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                       if col != target_col]
        
        if len(numeric_cols) > 0:
            st.subheader("Performance vs Other Factors")
            factor_col = st.selectbox("Select factor to compare", numeric_cols)
            
            if factor_col:
                fig3 = px.scatter(data, x=factor_col, y=target_col,
                                 title=f'{target_col} vs {factor_col}',
                                 trendline="ols")
                st.plotly_chart(fig3, use_container_width=True)
    
    def _create_demographic_analysis(self, data: pd.DataFrame):
        """Create demographic analysis"""
        categorical_cols = data.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            st.info("No categorical columns found for demographic analysis.")
            return
        
        # Demographics distribution
        col1, col2 = st.columns(2)
        
        for i, col in enumerate(categorical_cols[:4]):  # Show up to 4 categorical columns
            with col1 if i % 2 == 0 else col2:
                value_counts = data[col].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                            title=f'Distribution of {col}')
                st.plotly_chart(fig, use_container_width=True)
        
        # Demographic vs performance
        grade_cols = [col for col in data.columns 
                     if any(keyword in col.lower() for keyword in ['grade', 'score'])]
        
        if grade_cols and len(categorical_cols) > 0:
            st.subheader("Performance by Demographics")
            demo_col = st.selectbox("Select demographic factor", categorical_cols)
            
            if demo_col:
                fig = px.box(data, x=demo_col, y=grade_cols[0],
                            title=f'{grade_cols[0]} by {demo_col}')
                st.plotly_chart(fig, use_container_width=True)
    
    def _create_correlation_analysis(self, data: pd.DataFrame):
        """Create correlation analysis"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns for correlation analysis.")
            return
        
        # Correlation matrix
        corr_matrix = data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Correlation Matrix", 
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
        
        # Strong correlations
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corrs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_val
                    })