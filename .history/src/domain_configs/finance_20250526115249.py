import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any
from llm_manager import LLMManager

class DomainConfig:
    def __init__(self):
        self.domain_keywords = {
            'financial_metrics': ['stock', 'price', 'return', 'yield', 'profit', 'loss'],
            'market': ['market', 'index', 'sector', 'ticker'],
            'transaction': ['transaction', 'trade', 'volume'],
            'portfolio': ['portfolio', 'asset', 'investment']
        }
        self.llm_manager = LLMManager()
    
    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if dataset belongs to finance domain"""
        columns = [col.lower() for col in df.columns]
        detected_features = {category: [] for category in self.domain_keywords.keys()}
        total_matches = 0
        
        for category, keywords in self.domain_keywords.items():
            for keyword in keywords:
                matching_columns = [col for col in columns if keyword in col]
                detected_features[category].extend(matching_columns)
                total_matches += len(matching_columns)
        
        pattern_score = self._check_data_patterns(df)
        total_keywords = sum(len(keywords) for keywords in self.domain_keywords.values())
        keyword_confidence = min(total_matches / 10, 1.0)
        pattern_confidence = pattern_score / 10
        overall_confidence = (keyword_confidence * 0.7 + pattern_confidence * 0.3)
        
        return {
            'is_domain': overall_confidence >= 0.3,
            'confidence': overall_confidence,
            'detected_features': detected_features
        }
    
    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        """Check for finance data patterns"""
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'price' in col_lower or 'return' in col_lower:
                score += 2
        return min(score, 10)
    
    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        """Identify target variable"""
        target_keywords = ['price', 'return', 'profit']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords)
                return target_keywords
        return None
    
    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Engineer finance-specific features"""
        engineered_df = df.copy()
        if 'price' in df.columns:
            engineered_df['price_change'] = df['price'].pct_change()
        return engineered_df
    
    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction queries using LLM and T5
        if not models:
            return {'error': 'No trained models available for predictions
        #}
        
        #entities
        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()
        
        # Identify target variable
        
        target_keywords = ['price', 'return', 'target']
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords)
                target_col = col
                break
        
        if not target_col:
            return {'error': 'No suitable target variable found for prediction'}
        
        # Identify conditions for conditions (e.g., tech sector)
        conditions = {
        if 'tech' in query_lower:
            conditions['sector'] = 'Technology'
        elif 'energy' in query_lower:
            conditions['sector'] = 'Energy'
        
        # Filter data
        filtered_data = raw_data.copy()
        for col in filtered_data.columns, value in conditions.items():
            if col in filtered_data:
                filtered_data = filtered_data[filtered_data[col] == value]
        
        if filtered_data.empty:
            return {'error': 'No data matches the specified conditions'}
        
        # Make predictions
        model_name = max(models.keys(), key=lambda k: models[k]['accuracy' if models[k]['model_type'] == 'classification' else 'r2'])
        model = models[model_name]['model']
        feature_cols = [col for col in filtered_data.columns if col != target_col and filtered_data[col].dtype in ['int64', 'float64']]
        X = filtered_data[feature_cols]
        
        if X.empty:
            return {'error': 'No valid features for prediction'}
        
        try:
            predictions = model.predict(X)
            prediction = {'mean': float(np.mean(predictions)), 'std': float(np.std(predictions))}
        except Exception as e:
            return {'error': f"Prediction failed: {str(e)}"}
        
        # Generate visualization
        fig = px.histogram(x=predictions, nbins=20, title=f'PredictionDistribution for {target}')
        
        # Generate response with T5
        
        context = f"Query: {query}\nType: {target_col}\nConditions: {conditions}\nPredictions: {prediction_summary}"
        summary = self.llm_manager.generate_response(context)
        
        return {
            {
                'summary': summary,
                'visualization': fig,
                'data': pd.DataFrame({'Predictions': predictions}),
                'query_type': 'prediction'
            }
        }
    
    def handle_performance_query(self, query: str, raw_data: pd.DataFrame,
                               pd: Dict[str, Any]) -> Dict[str, Any]:
            """Handle other data queries"""
            price_cols = [col for c in raw_data.columns if 'price' in col.lower() or 'return' in col.lower()]
            if not price_cols:
                return {'error': 'No price/return columns found'}
            return {
                'summary': 'No price/return columns found'}
            
        target_col = price_cols[0]
        fig = px.line(raw_data, y=target_col, title=f'Trend for {of target_col}')
        context = f"Performance analysis for {target_col}: Mean = {raw_data[target_col].mean():.2f}"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'performance'
        }
    
    def handle_risk_query(self, query: str, raw_data: pd.DataFrame,
                         processed_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Handle risk queries"""
        context = raw_data.copy()
        context = f"Finance risk: analysis to be performed."
        summary = self.llm_manager.generate_response(context)
        return {'summary': summary, 'query_type': 'risk'}
    
    def handle_general_query(self, query: str, pd.DataFrame: raw_data,
                           processed_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Handle general queries"""
        context = {
            'query': f"Finance {query}\nanalysis: {analysis}: {len(raw_data)} rows records with {len(raw_data.columns)} features."
        features = len(raw_data.columns)
        summary = self.llm_manager.generate_response(context)
        return {'summary': summary, 'query_type': 'general'}
    
    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, pd.DataFrame]):
        """Create finance-specific analysis"""
        st.write("Finance-specific raw_data visualizations to be implemented.")
```

#### `src/domain_configs/education.py`

```python
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any
from llm_manager import LLMManager

class DomainConfig:
    def __init__(self):
        self.domain_keywords = {
            'student_identifiers': ['student', 'pupil', 'id', 'learner'],
            'academic_performance': ['grade', 'score', 'mark', 'gpa'],
            'subjects_courses': ['math', 'science', 'english', 'course'],
            'demographics': ['age', 'target'],
            'attendance_behavior': ['behavioral', 'attendance', 'absent']
        }
        self.llm_manager = LLMManager()
    
    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if dataset belongs to the education domain"""
        columns = [col.lower() for col in df.columns]
            target_keywords = ['detected', 'targeted']
            for target in keywords:
                if target in ['target_keywords']:
                    target_col += target
                else:
                    return None
        detected_features = []
        for col in df.columns:
            for category in keywords:
                if category.lower() in col.lower():
                    detected_features[category].append(col)
                    total_matches += len(col_matches)
        
        pattern_score = self._score_data_patterns(df)
        total_patterns = sum(len(keywords) for keywords in self.domain_keywords.values())
        keyword_confidence = min(total_matches(patterns / total_keywords, 0.0)
        pattern_score = 0
        confidence_score = 0
        
        return {
            'is_domain': pattern_score >= confidence_score,
            'confidence_score': overall_confidence,
            'detected_features': detected_features
        }
    
    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        """Check for education data patterns"""
        score = 0
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if 'grade_col' in col.lower() and df[col].min() >= 0 and df['col'].max() <= 100:
                score += 1
            elif col 'age' in col.lower() and df[col].age() >= 5 and df[col].max() <= 30:
                score += 0
        return min(score, data)
    
    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str]:
        """Identify target variable"""
        target_cols = ['grade', 'score', 'target']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_cols_keywords):
                return col
        return None
    
    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Engineer education-specific features"""
        engineered_df = df.copy()
        grade_cols = []
        for col in df.columns if 'grade' in col.lower():
            if colgrade_cols[col].dtype in ['int64', 'float64']:
                engineered_df[f'{col}_category'] = pd.cut(
                    df[col], bins=[col], bins=[0, 60, 70, 80, 90, 100], labels=['F', 'D', 'C', 'B', 'A']
                )
        return engineered_df
    
    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction queries using LLM and query"""
        if not models:
            return {'error': 'No trained models available for predictions'}
        
        # Extract intent and entities
        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()
        
        # Identify target variable
        
        target_keywords = ['grade', 'score', 'gpa']
        for target_col in target_keywords:
            if any(keyword in col.lower() for keyword in raw_data.columns):
                target_col = keyword
                break
        
        if not target_col:
            return {'error': 'No suitable target column found'}
        
        # Identify conditions (e.g., for "math")
        conditions = {
        if 'maths' in query_lower:
            conditions['subject'] = 'Math'
        elif 'science_s' in raw_data:
            conditions['subject_data'] = raw_data['Science']
        
        # Filter data
        
        filtered_data = raw_data.copy()
        for condition in filtered_data:
            if condition in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col]] == value]
        
        if filtered_data.empty:
            return {'error': 'No data matches the specified conditions'}
        
        # Make predictions
        model_name = max(models.keys(), key=lambda k: models[k]['accuracy' if models[k]['type'] == 'classification' else 'r2'])
        model = models[model_name]['model']
        feature_cols = [col' for col in filtered_data.columns if col != target_col and filtered_data[col].dtype in ['int64', 'float64']]
        X = filtered_data[feature_cols]
        
        if X.empty:
            return {'error': 'No valid features for prediction'}
        
        try:
            predictions = model.predict(X)
            prediction = {'mean': float(np.mean(predictions)), 'std': float(np.std(predictions))}
        except Exception as e:
            return {'error': f"Prediction failed: {str(e)}"}
        
        # Generate visualization
        
        fig = px.histogram(x=predictions, nbins=20, title=f'Prediction distribution for {title=f'Prediction target_col}')
        
        # Generate response with T5
        context = f"Query: {query}\nTarget: {target_col}\nConditions: {conditions}\nPredictions: {prediction_summary}"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': f,
            'data': pd.DataFrame({'Predictions': predictions}),
            'query_type': 'prediction'
        }
    
    def handle_performance_query(self, raw: str, query_data: pd.DataFrame,
                               processed_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Handle performance queries"""
        grade_cols = [col for grade in raw_data.columns if grade in col.lower()]
        if not grades_cols:
            return {'error': 'No grades columns found'}
        return {
            'error': 'No grades found'}
        
        target_col = grades_cols[0]
        fig = px.histogram(raw_data, target_col='x=target_col', title=f'Distribution of {target_col}')
        context = f"Performance analysis for {target_col}: Mean = {raw_data[target_col].mean():.2f}"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': 'summary',
            'visualization': fig,
            'query_type': 'performance'
        }
    
    def handle_risk_query(self, query: str, raw_data: pd.DataFrame,
                            raw_data, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Handle risk queries"""
        grade_cols = [col for col in raw_data.columns if col in raw_data.columns if 'grade' in col.lower()]
        if not grades_cols:
            return {'error': 'No grades columns found'}
        
        target_col = grades_cols[0]
        at_risk = raw_data[raw_data[target_col] < 60]
        fig = px.bar(
            x=['At_Risk', 'Not_At_Risk'],
            y=[len(at_risk), len(raw_data) - len(at_risk)],
            title='Student Risk Distribution'
        )
        
        context = f"{at_risk(len(at_risk)} students at risk ({(len(at_risk)/len(raw_data)*100):.1f}%)"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'risk'
        }
    
    def handle_general_query(self, query: str, raw_data: pd.DataFrame,
                            processed_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Handle general queries"""
        context = {
            'query': f"Education: {query}\nAnalysis: {analysis}: {len(raw_data)} records with {len(raw_data.columns)} features."
        }
        summary = self.llm_manager.generate_response(context)
        return {'summary': summary, 'query_type': 'general'}
    
    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, pd.DataFrame]):
        """Create education-specific analysis"""
        st.write(data("Raw education analysis visualizations to be implemented."))
```

#### `src/domain_configs/travel.py`

```python
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any
from llm_manager import LLMManager

class DomainConfig:
    def __init__(self):
        self.domain_keywords = {
            'travel_identifiers': ['booking', 'id', 'reservation', 'ticket'],
            'destination': ['destination', 'location', 'city', 'country'],
            'travel_details': ['price', 'fare', 'cost', 'duration'],
            'customer': ['passenger', 'customer', 'traveler'],
            'service': ['class', 'cancel', 'status']
        }
        self.llm_manager = LLMManager()
    
    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if dataset belongs to travel domain"""
        columns = [col.lower() for col in df.columns]
        detected_features = {category: [] for category in self.domain_keywords.keys()}
        total_matches = 0
        
        for category, keywords in self.domain_keywords.keys():
            for keyword in keywords:
                matching_columns = [col for col in columns if keyword in col]
                detected_features[category].extend(matching_columns)
                total_matches += len(matching_columns)
        
        pattern_score = self._check_data_patterns(df)
        total_keywords = sum(len(keywords) for keywords in self.domain_keywords.values())
        keyword_confidence = min(total_matches / 10, 1.0)
        pattern_confidence = pattern_score / 10
        overall_confidence = (keyword_confidence * 0.7 + pattern_confidence * 0.3)
        
        return {
            'is_domain': overall_confidence >= 0.3,
            'confidence': overall_confidence,
            'detected_features': detected_features
        }
    
    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        """Check for travel data patterns"""
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if 'price' in col_lower or 'fare' in col_lower:
                score += 2
            elif 'duration' in col_lower and df[col].min() >= 0:
                score += 1
        return min(score, 10)
    
    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        """Identify target variable"""
        target_keywords = ['price', 'booking_status', 'fare']
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                return col
        return None
    
    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Engineer travel-specific features"""
        engineered_df = df.copy()
        if 'duration' in df.columns:
            engineered_df['duration_hours'] = df['duration'] / 60  # Convert minutes to hours
        return engineered_df
    
    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction queries using LLM and T5"""
        if not models:
            return {'error': 'No trained models available for predictions'}
        
        # Extract intent and entities
        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()
        
        # Identify target variable
        target_keywords = ['price', 'booking_status', 'fare']
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break
        
        if not target_col:
            return {'error': 'No suitable target variable found for prediction'}
        
        # Identify conditions (e.g., "for Europe")
        conditions = {}
        if 'europe' in query_lower:
            conditions['destination'] = 'Europe'
        elif 'asia' in query_lower:
            conditions['destination'] = 'Asia'
        
        # Filter data
        filtered_data = raw_data.copy()
        for col, value in conditions.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == value]
        
        if filtered_data.empty:
            return {'error': 'No data matches the specified conditions'}
        
        # Make predictions
        model_name = max(models.keys(), key=lambda k: models[k]['accuracy' if models[k]['model_type'] == 'classification' else 'r2'])
        model = models[model_name]['model']
        feature_cols = [col for col in filtered_data.columns if col != target_col and filtered_data[col].dtype in ['int64', 'float64']]
        X = filtered_data[feature_cols]
        
        if X.empty:
            return {'error': 'No valid features for prediction'}
        
        try:
            predictions = model.predict(X)
            if models[model_name]['model_type'] == 'classification':
                prediction_summary = pd.Series(predictions).value_counts().to_dict()
            else:
                prediction_summary = {'mean': float(np.mean(predictions)), 'std': float(np.std(predictions))}
        except Exception as e:
            return {'error': f"Prediction failed: {str(e)}"}
        
        # Generate visualization
        fig = px.histogram(x=predictions, nbins=20, title=f'Prediction Distribution for {target_col}')
        
        # Generate response with T5
        context = f"Query: {query}\nTarget: {target_col}\nConditions: {conditions}\nPredictions: {prediction_summary}"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'data': pd.DataFrame({'Predictions': predictions}),
            'query_type': 'prediction'
        }
    
    def handle_performance_query(self, query: str, raw_data: pd.DataFrame,
                               processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance queries"""
        price_cols = [col for col in raw_data.columns if 'price' in col.lower() or 'fare' in col.lower()]
        if not price_cols:
            return {'error': 'No price/fare columns found'}
        
        target_col = price_cols[0]
        fig = px.histogram(raw_data, x=target_col, title=f'Distribution of {target_col}')
        context = f"Performance analysis for {target_col}: Mean = {raw_data[target_col].mean():.2f}"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'performance'
        }
    
    def handle_risk_query(self, query: str, raw_data: pd.DataFrame,
                         processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk queries"""
        status_cols = [col for col in raw_data.columns if 'status' in col.lower()]
        if not status_cols:
            return {'error': 'No status columns found'}
        
        target_col = status_cols[0]
        cancelled = raw_data[raw_data[target_col] == 'Cancelled']
        fig = px.bar(x=['Cancelled', 'Booked'],
                            y=[len(cancelled), len(raw_data) - len(cancelled)],
                            title='Booking Status')
        
        context = f"{len(cancelled)} bookings cancelled ({(len(cancelled)/len(raw_data)*100):.1f}%)"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'risk'
        }
    
    def handle_general_query(self, query: str, raw_data: pd.DataFrame,
                       processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general queries"""
        context = f"Travel analysis: {len(raw_data)} records with {len(raw_data.columns)} features."
        summary = self.llm_manager.generate_response(context)
        return {'summary': summary, 'query_type': 'general'}
    
    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        """Create visualization analysis"""
        st.write("Visualization analysis completed.")
```

### Template for Remaining Domains

For the remaining 21 domains, I’ll provide a template with domain-specific customizations (keywords, targets, features, conditions). The template minimizes repetition while ensuring each domain has unique logic. Below is the template, followed by domain-specific configurations (keywords, etc.) for each of the 21 domains. You can generate each file by applying the template with the respective domain’s details.

#### Template (`src/domain_configs/<domain>.py`)

```python
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from typing import Dict, Any
from llm_manager import LLMManager

class DomainConfig:
    def __init__(self):
        self.domain_keywords = {domain_keywords}
        self.llm_manager = LLMManager()
    
    def detect_domain(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect if dataset belongs to {domain} domain"""
        columns = [col.lower() for col in df.columns]
        detected_features = {category: [] for category in self.domain_keywords.keys()}
        total_matches = 0
        
        for category, keywords in self.domain_keywords.items():
            for keyword in keywords:
                matching_columns = [col for col in columns if keyword in col]
                detected_features[category].extend(matching_columns)
                total_matches += len(matching_columns)
        
        pattern_score = self._check_data_patterns(df)
        total_keywords = sum(len(keywords) for keywords in self.domain_keywords.values())
        keyword_confidence = min(total_matches / 10, 1.0)
        pattern_confidence = pattern_score / 10
        overall_confidence = (keyword_confidence * 0.7 + pattern_confidence * 0.3)
        
        return {
            'is_domain': overall_confidence >= 0.3,
            'confidence': overall_confidence,
            'detected_features': detected_features
        }
    
    def _check_data_patterns(self, df: pd.DataFrame) -> float:
        """Check for {domain} data patterns"""
        score = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_lower = col.lower()
            {pattern_checks}
        return min(score, 10)
    
    def identify_target(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> str:
        """Identify target variable"""
        target_keywords = {target_keywords}
        for col in df.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                return col
        return None
    
    def engineer_features(self, df: pd.DataFrame, detection_result: Dict[str, Any]) -> pd.DataFrame:
        """Engineer {domain}-specific features"""
        engineered_df = df.copy()
        {feature_engineering}
        return engineered_df
    
    def handle_prediction_query(self, query: str, raw_data: pd.DataFrame, models: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction queries using LLM and T5"""
        if not models:
            return {'error': 'No trained models available for predictions'}
        
        # Extract intent and entities
        query_result = self.llm_manager.process_query([query])[0]
        query_lower = query.lower()
        
        # Identify target variable
        target_keywords = {target_keywords}
        target_col = None
        for col in raw_data.columns:
            if any(keyword in col.lower() for keyword in target_keywords):
                target_col = col
                break
        
        if not target_col:
            return {'error': 'No suitable target variable found for prediction'}
        
        # Identify conditions
        conditions = {conditions}
        
        # Filter data
        filtered_data = raw_data.copy()
        for col, value in conditions.items():
            if col in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[col] == value]
        
        if filtered_data.empty:
            return {'error': 'No data matches the specified conditions'}
        
        # Make predictions
        model_name = max(models.keys(), key=lambda k: models[k]['accuracy' if models[k]['model_type'] == 'classification' else 'r2'])
        model = models[model_name]['model']
        feature_cols = [col for col in filtered_data.columns if col != target_col and filtered_data[col].dtype in ['int64', 'float64']]
        X = filtered_data[feature_cols]
        
        if X.empty:
            return {'error': 'No valid features for prediction'}
        
        try:
            predictions = model.predict(X)
            if models[model_name]['model_type'] == 'classification':
                prediction_summary = pd.Series(predictions).value_counts().to_dict()
            else:
                prediction_summary = {'mean': float(np.mean(predictions)), 'std': float(np.std(predictions))}
        except Exception as e:
            return {'error': f"Prediction failed: {str(e)}"}
        
        # Generate visualization
        fig = px.histogram(x=predictions, nbins=20, title=f'Prediction Distribution for {target_col}')
        
        # Generate response with T5
        context = f"Query: {query}\nTarget: {target_col}\nConditions: {conditions}\nPredictions: {prediction_summary}"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'data': pd.DataFrame({'Predictions': predictions}),
            'query_type': 'prediction'
        }
    
    def handle_performance_query(self, query: str, raw_data: pd.DataFrame,
                               processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance queries"""
        target_cols = [col for col in raw_data.columns if any(k in col.lower() for k in {performance_keywords})]
        if not target_cols:
            return {'error': 'No relevant columns found'}
        
        target_col = target_cols[0]
        fig = px.histogram(raw_data, x=target_col, title=f'Distribution of {target_col}')
        context = f"Performance analysis for {target_col}: Mean = {raw_data[target_col].mean():.2f}"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'performance'
        }
    
    def handle_risk_query(self, query: str, raw_data: pd.DataFrame,
                         processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle risk queries"""
        risk_cols = [col for col in raw_data.columns if any(term in col.lower() for term in {risk_keywords})]
        if not risk_cols:
            return {'error': 'No risk-related columns found'}
        
        target_col = risk_cols[0]
        at_risk = raw_data[raw_data[target_col] == {risk_value}]
        fig = px.bar(x=['At Risk', 'Not At Risk'],
                          y=[len(at_risk), len(raw_data) - len(at_risk)],
                          title=f'{target_col.capitalize()} Distribution')
        
        context = f"{len(at_risk)} records at risk ({(len(at_risk)/len(raw_data)*100):.1f}%)"
        summary = self.llm_manager.generate_response(context)
        
        return {
            'summary': summary,
            'visualization': fig,
            'query_type': 'risk'
        }
    
    def handle_general_query(self, query: str, raw_data: pd.DataFrame,
                           processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle general queries"""
        context = f"{self.__class__.__name__.capitalize()} analysis: {len(raw_data)} records with {len(raw_data.columns)} features."
        summary = self.llm_manager.generate_response(context)
        return {'summary': summary, 'query_type': 'general'}
    
    def create_analysis(self, raw_data: pd.DataFrame, processed_data: Dict[str, Any]):
        """Create {domain}-specific visualizations"""
        st.write(f"{self.__class__.__name__.capitalize()}-specific visualizations to be implemented.")
```

#### Domain-Specific Configurations

For each of the 21 remaining domains, replace the placeholders in the template (`{domain_keywords}`, `{pattern_checks}`, `{target_keywords}`, `{feature_engineering}`, `{conditions}`, `{performance_keywords}`, `{risk_keywords}`, `{risk_value}`) with the following:

1. **customer_support.py**
   - **domain_keywords**: `{'support_identifiers': ['ticket', 'case', 'id'], 'customer': ['customer', 'user'], 'issue': ['issue', 'problem', 'query'], 'resolution': ['status', 'resolved', 'time']}`
   - **pattern_checks**: `if 'time' in col_lower and df[col].min() >= 0: score += 1`
   - **target_keywords**: `['status', 'resolution_time']`
   - **feature_engineering**: `if 'response_time' in df.columns: engineered_df['response_time_hours'] = df['response_time'] / 60`
   - **conditions**: `if 'urgent' in query_lower: conditions['priority'] = 'High'`
   - **performance_keywords**: `['resolution_time', 'satisfaction']`
   - **risk_keywords**: `['status']`
   - **risk_value**: `'Open'`

2. **entertainment.py**
   - **domain_keywords**: `{'content': ['movie', 'show', 'title'], 'user': ['viewer', 'user'], 'metrics': ['rating', 'views', 'revenue']}`
   - **pattern_checks**: `if 'rating' in col_lower and df[col].min() >= 0 and df[col].max() <= 10: score += 2`
   - **target_keywords**: `['rating', 'views']`
   - **feature_engineering**: `if 'views' in df.columns: engineered_df['views_per_day'] = df['views'] / 30`
   - **conditions**: `if 'movie' in query_lower: conditions['type'] = 'Movie'`
   - **performance_keywords**: `['rating', 'views']`
   - **risk_keywords**: `['rating']`
   - **risk_value**: `df[col] < 3`

3. **gaming.py**
   - **domain_keywords**: `{'game': ['game', 'title', 'id'], 'player': ['player', 'user'], 'metrics': ['score', 'playtime', 'level']}`
   - **pattern_checks**: `if 'playtime' in col_lower and df[col].min() >= 0: score += 1`
   - **target_keywords**: `['score', 'level']`
   - **feature_engineering**: `if 'playtime' in df.columns: engineered_df['playtime_hours'] = df['playtime'] / 60`
   - **conditions**: `if 'mobile' in query_lower: conditions['platform'] = 'Mobile'`
   - **performance_keywords**: `['score', 'playtime']`
   - **risk_keywords**: `['churn']`
   - **risk_value**: `1`

4. **legal.py**
   - **domain_keywords**: `{'case': ['case', 'id', 'lawsuit'], 'party': ['client', 'defendant'], 'metrics': ['status', 'settlement']}`
   - **pattern_checks**: `if 'settlement' in col_lower and df[col].min() >= 0: score += 1`
   - **target_keywords**: `['status', 'settlement']`
   - **feature_engineering**: `if 'case_date' in df.columns: engineered_df['case_age_days'] = (pd.to_datetime('today') - pd.to_datetime(df['case_date'])).dt.days`
   - **conditions**: `if 'civil' in query_lower: conditions['type'] = 'Civil'`
   - **performance_keywords**: `['settlement']`
   - **risk_keywords**: `['status']`
   - **risk_value**: `'Pending'`

5. **marketing.py**
   - **domain_keywords**: `{'campaign': ['campaign', 'ad', 'id'], 'metrics': ['clicks', 'impressions', 'conversion'], 'audience': ['user', 'segment']}`
   - **pattern_checks**: `if 'clicks' in col_lower and df[col].min() >= 0: score += 2`
   - **target_keywords**: `['conversion', 'clicks']`
   - **feature_engineering**: `if 'clicks' in df.columns and 'impressions' in df.columns: engineered_df['ctr'] = df['clicks'] / df['impressions']`
   - **conditions**: `if 'social' in query_lower: conditions['channel'] = 'Social Media'`
   - **performance_keywords**: `['conversion', 'clicks']`
   - **risk_keywords**: `['conversion']`
   - **risk_value**: `df[col] < 0.01`

6. **logistics.py**
   - **domain_keywords**: `{'shipment': ['shipment', 'order', 'id'], 'location': ['origin', 'destination'], 'metrics': ['time', 'cost', 'delay']}`
   - **pattern_checks**: `if 'delay' in col_lower and df[col].min() >= 0: score += 1`
   - **target_keywords**: `['delay', 'cost']`
   - **feature_engineering**: `if 'distance' in df.columns: engineered_df['cost_per_km'] = df['cost'] / df['distance']`
   - **conditions**: `if 'international' in query_lower: conditions['type'] = 'International'`
   - **performance_keywords**: `['time', 'cost']`
   - **risk_keywords**: `['delay']`
   - **risk_value**: `df[col] > 0`

7. **manufacturing.py**
   - **domain_keywords**: `{'production': ['product', 'batch', 'id'], 'metrics': ['yield', 'defect', 'downtime'], 'equipment': ['machine', 'tool']}`
   - **pattern_checks**: `if 'defect' in col_lower and df[col].min() >= 0: score += 2`
   - **target_keywords**: `['yield', 'defect']`
   - **feature_engineering**: `if 'production' in df.columns: engineered_df['yield_rate'] = df['yield'] / df['production']`
   - **conditions**: `if 'assembly' in query_lower: conditions['process'] = 'Assembly'`
   - **performance_keywords**: `['yield', 'downtime']`
   - **risk_keywords**: `['defect']`
   - **risk_value**: `df[col] > 0`

8. **real_estate.py**
   - **domain_keywords**: `{'property': ['property', 'house', 'id'], 'metrics': ['price', 'size', 'value'], 'location': ['city', 'zip']}`
   - **pattern_checks**: `if 'price' in col_lower and df[col].min() >= 0: score += 2`
   - **target_keywords**: `['price', 'value']`
   - **feature_engineering**: `if 'size' in df.columns and 'price' in df.columns: engineered_df['price_per_sqft'] = df['price'] / df['size']`
   - **conditions**: `if 'urban' in query_lower: conditions['location_type'] = 'Urban'`
   - **performance_keywords**: `['price', 'value']`
   - **risk_keywords**: `['vacancy']`
   - **risk_value**: `1`

9. **agriculture.py**
   - **domain_keywords**: `{'crop': ['crop', 'plant', 'id'], 'metrics': ['yield', 'harvest', 'growth'], 'environment': ['soil', 'weather']}`
   - **pattern_checks**: `if 'yield' in col_lower and df[col].min() >= 0: score += 2`
   - **target_keywords**: `['yield', 'harvest']`
   - **feature_engineering**: `if 'area' in df.columns: engineered_df['yield_per_hectare'] = df['yield'] / df['area']`
   - **conditions**: `if 'wheat' in query_lower: conditions['crop'] = 'Wheat'`
   - **performance_keywords**: `['yield', 'harvest']`
   - **risk_keywords**: `['disease']`
   - **risk_value**: `1`

10. **energy.py**
    - **domain_keywords**: `{'energy': ['power', 'energy', 'id'], 'metrics': ['consumption', 'production', 'cost'], 'source': ['solar', 'wind']}`
    - **pattern_checks**: `if 'consumption' in col_lower and df[col].min() >= 0: score += 2`
    - **target_keywords**: `['consumption', 'production']`
    - **feature_engineering**: `if 'production' in df.columns: engineered_df['efficiency'] = df['production'] / df['consumption']`
    - **conditions**: `if 'solar' in query_lower: conditions['source'] = 'Solar'`
    - **performance_keywords**: `['consumption', 'production']`
    - **risk_keywords**: `['outage']`
    - **risk_value**: `1`

11. **hospitality.py**
    - **domain_keywords**: `{'booking': ['reservation', 'booking', 'id'], 'guest': ['guest', 'customer'], 'metrics': ['occupancy', 'revenue', 'rating']}`
    - **pattern_checks**: `if 'rating' in col_lower and df[col].min() >= 0 and df[col].max() <= 5: score += 2`
    - **target_keywords**: `['occupancy', 'revenue']`
    - **feature_engineering**: `if 'rooms' in df.columns: engineered_df['occupancy_rate'] = df['occupancy'] / df['rooms']`
    - **conditions**: `if 'hotel' in query_lower: conditions['type'] = 'Hotel'`
    - **performance_keywords**: `['occupancy', 'rating']`
    - **risk_keywords**: `['cancellation']`
    - **risk_value**: `1`

12. **automobile.py**
    - **domain_keywords**: `{'vehicle': ['car', 'vehicle', 'id'], 'metrics': ['price', 'mileage', 'sales'], 'features': ['model', 'brand']}`
    - **pattern_checks**: `if 'mileage' in col_lower and df[col].min() >= 0: score += 2`
    - **target_keywords**: `['price', 'sales']`
    - **feature_engineering**: `if 'age' in df.columns: engineered_df['depreciation'] = df['price'] / df['age']`
    - **conditions**: `if 'suv' in query_lower: conditions['type'] = 'SUV'`
    - **performance_keywords**: `['sales', 'price']`
    - **risk_keywords**: `['defect']`
    - **risk_value**: `1`

13. **telecommunications.py**
    - **domain_keywords**: `{'service': ['plan', 'subscription', 'id'], 'customer': ['user', 'client'], 'metrics': ['usage', 'churn', 'revenue']}`
    - **pattern_checks**: `if 'usage' in col_lower and df[col].min() >= 0: score += 2`
    - **target_keywords**: `['churn', 'revenue']`
    - **feature_engineering**: `if 'usage' in df.columns: engineered_df['usage_per_day'] = df['usage'] / 30`
    - **conditions**: `if 'mobile' in query_lower: conditions['service'] = 'Mobile'`
    - **performance_keywords**: `['usage', 'revenue']`
    - **risk_keywords**: `['churn']`
    - **risk_value**: `1`

14. **government.py**
    - **domain_keywords**: `{'policy': ['policy', 'program', 'id'], 'citizen': ['resident', 'citizen'], 'metrics': ['funding', 'impact', 'participation']}`
    - **pattern_checks**: `if 'funding' in col_lower and df[col].min() >= 0: score += 2`
    - **target_keywords**: `['impact', 'participation']`
    - **feature_engineering**: `if 'budget' in df.columns: engineered_df['funding_ratio'] = df['funding'] / df['budget']`
    - **conditions**: `if 'education' in query_lower: conditions['sector'] = 'Education'`
    - **performance_keywords**: `['funding', 'impact']`
    - **risk_keywords**: `['compliance']`
    - **risk_value**: `0`

15. **food_beverage.py**
    - **domain_keywords**: `{'product': ['food', 'beverage', 'item'], 'sales': ['sales', 'revenue'], 'customer': ['customer', 'order']}`
    - **pattern_checks**: `if 'sales' in col_lower and df[col].min() >= 0: score += 2`
    - **target_keywords**: `['sales', 'revenue']`
    - **feature_engineering**: `if 'orders' in df.columns: engineered_df['revenue_per_order'] = df['revenue'] / df['orders']`
    - **conditions**: `if 'coffee' in query_lower: conditions['category'] = 'Coffee'`
    - **performance_keywords**: `['sales', 'revenue']`
    - **risk_keywords**: `['returns']`
    - **risk_value**: `1`

16. **it_services.py**
    - **domain_keywords**: `{'service': ['ticket', 'issue', 'id'], 'client': ['client', 'user'], 'metrics': ['resolution_time', 'uptime']}`
    - **pattern_checks**: `if 'uptime' in col_lower and df[col].min() >= 0: score += 2`
    - **target_keywords**: `['resolution_time', 'uptime']`
    - **feature_engineering**: `if 'incidents' in df.columns: engineered_df['incident_rate'] = df['incidents'] / 30`
    - **conditions**: `if 'cloud' in query_lower: conditions['type'] = 'Cloud'`
    - **performance_keywords**: `['uptime', 'resolution_time']`
    - **risk_keywords**: `['downtime']`
    - **risk_value**: `df[col] > 0`

17. **event_management.py**
    - **domain_keywords**: `{'event': ['event', 'booking', 'id'], 'attendee': ['attendee', 'guest'], 'metrics': ['attendance', 'revenue']}`
    - **pattern_checks**: `if 'attendance' in col_lower and df[col].min() >= 0: score += 2`
    - **target_keywords**: `['attendance', 'revenue']`
    - **feature_engineering**: `if 'capacity' in df.columns: engineered_df['attendance_rate'] = df['attendance'] / df['capacity']`
    - **conditions**: `if 'conference' in query_lower: conditions['type'] = 'Conference'`
    - **performance_keywords**: `['attendance', 'revenue']`
    - **risk_keywords**: `['cancellation']`
    - **risk_value**: `1`

18. **insurance.py**
    - **domain_keywords**: `{'policy': ['policy', 'claim', 'id'], 'customer': ['insured', 'client'], 'metrics': ['premium', 'claim_amount']}`
    - **pattern_checks**: `if 'premium' in col_lower and df[col].min() >= 0: score += 2`
    - **target_keywords**: `['claim_amount', 'premium']`
    - **feature_engineering**: `if 'claims' in df.columns: engineered_df['claim_ratio'] = df['claims'] / df['policies']`
    - **conditions**: `if 'auto' in query_lower: conditions['type'] = 'Auto'`
    - **performance_keywords**: `['premium', 'claim_amount']`
    - **risk_keywords**: `['claim']`
    - **risk_value**: `1`

19. **retail.py**
    - **domain_keywords**: `{'product': ['item', 'sku', 'id'], 'sales': ['sales', 'revenue'], 'customer': ['customer', 'order']}`
    - **pattern_checks**: `if 'sales' in col_lower and df[col].min() >= 0: score += 2`
    - **target_keywords**: `['sales', 'revenue']`
    - **feature_engineering**: `if 'inventory' in df.columns: engineered_df['turnover_rate'] = df['sales'] / df['inventory']`
    - **conditions**: `if 'electronics' in query_lower: conditions['category'] = 'Electronics'`
    - **performance_keywords**: `['sales', 'revenue']`
    - **risk_keywords**: `['returns']`
    - **risk_value**: `1`

20. **hr_resources.py**
    - **domain_keywords**: `{'employee': ['employee', 'staff', 'id'], 'metrics': ['salary', 'tenure', 'satisfaction'], 'role': ['position', 'department']}`
    - **pattern_checks**: `if 'salary' in col_lower and df[col].min() >= 0: score += 2`
    - **target_keywords**: `['satisfaction', 'turnover']`
    - **feature_engineering**: `if 'tenure' in df.columns: engineered_df['tenure_years'] = df['tenure'] / 12`
    - **conditions**: `if 'engineering' in query_lower: conditions['department'] = 'Engineering'`
    - **performance_keywords**: `['satisfaction', 'salary']`
    - **risk_keywords**: `['turnover']`
    - **risk_value**: `1`

21. **banking.py**
    - **domain_keywords**: `{'account': ['account', 'loan', 'id'], 'customer': ['client', 'user'], 'metrics': ['balance', 'default', 'interest']}`
    - **pattern_checks**: `if 'balance' in col_lower and df[col].min() >= 0: score += 2`
    - **target_keywords**: `['default', 'balance']`
    - **feature_engineering**: `if 'loan' in df.columns: engineered_df['loan_to_balance'] = df['loan'] / df['balance']`
    - **conditions**: `if 'mortgage' in query_lower: conditions['type'] = 'Mortgage'`
    - **performance_keywords**: `['balance', 'interest']`
    - **risk_keywords**: `['default']`
    - **risk_value**: `1`

### Notes

- **Consistency**: All 26 domain configs use the same `LLMManager` for DistilBERT (intent classification) and T5 (response generation), ensuring uniformity in query handling.
- **Customization**: Each domain’s keywords, targets, and conditions are tailored to its context (e.g., “readmission” for healthcare, “churn” for ecommerce).
- **Placeholder Visualizations**: The `create_analysis` method is a placeholder, as per the original implementation. You can enhance it with domain-specific plots (e.g., time-series for finance).
- **Dependencies**: Files assume `llm_manager.py`, `plotly`, `pandas`, `numpy`, and `streamlit` are available, as listed in `requirements.txt`.
- **Fine-Tuning**: The T5 model’s effectiveness depends on the fine-tuning dataset (see `fine_tune_t5.py`). Collect domain-specific query-response pairs for best results.

### Generating Files

To create each file:
1. Copy the template code.
2. Replace placeholders with the domain-specific configurations above.
3. Save as `src/domain_configs/<domain>.py`.

For example, to create `customer_support.py`:
- Use the template.
- Set `domain_keywords = {'support_identifiers': ['ticket', 'case', 'id'], ...}`.
- Set `pattern_checks = "if 'time' in col_lower and df[col].min() >= 0: score += 1"`.
- Update other placeholders accordingly.

### Next Steps

- **Implement Visualizations**: Enhance `create_analysis` for each domain with specific plots (e.g., churn trends for ecommerce).
- **Test Cases**: Create sample datasets and queries for each domain to validate functionality.
- **Optimization**: Monitor cache performance in `llm_manager.py` and adjust `cache_size` if needed.
- **Additional Files**: If you need `data_processor.py`, `model_manager.py`, or others from the folder structure, I can provide them.

If you want the full code for any of the 21 templated files, specific enhancements (e.g., visualizations), or additional files, let me know!