# src/utils/query_handler.py
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime
import json
import hashlib
import os
import re
import uuid
from utils.logger import Logger
from utils.llm_manager import LLMManager


class KnowledgeBaseHandler:
    """Handles knowledge-based responses for different query types."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.knowledge_base = self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self) -> Dict[str, Dict[str, Any]]:
        """Initialize knowledge base with domain-specific information."""
        return {
            'prediction': {
                'techniques': [
                    'Time series forecasting using ARIMA, LSTM, or Prophet',
                    'Regression models (Linear, Polynomial, Random Forest)',
                    'Machine learning approaches (XGBoost, Neural Networks)',
                    'Statistical methods and trend analysis'
                ],
                'factors': [
                    'Historical data patterns and seasonality',
                    'External variables and market conditions',
                    'Data quality and completeness',
                    'Model validation and cross-validation'
                ],
                'best_practices': [
                    'Use multiple models and ensemble methods',
                    'Validate predictions with out-of-sample data',
                    'Consider uncertainty intervals',
                    'Regular model retraining and monitoring'
                ]
            },
            'performance': {
                'metrics': {
                    'classification': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
                    'regression': ['MAE', 'RMSE', 'R²', 'MAPE', 'Adjusted R²'],
                    'clustering': ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']
                },
                'evaluation_methods': [
                    'Cross-validation (K-fold, Stratified)',
                    'Hold-out validation (Train/Test split)',
                    'Time series validation (Forward chaining)',
                    'Bootstrap sampling'
                ],
                'improvement_strategies': [
                    'Feature engineering and selection',
                    'Hyperparameter tuning',
                    'Model ensemble techniques',
                    'Data augmentation and preprocessing'
                ]
            },
            'risk': {
                'types': [
                    'Model risk (overfitting, underfitting)',
                    'Data risk (quality, bias, drift)',
                    'Operational risk (deployment, monitoring)',
                    'Regulatory and compliance risk'
                ],
                'assessment_methods': [
                    'Statistical significance testing',
                    'Sensitivity analysis',
                    'Stress testing scenarios',
                    'Monte Carlo simulation'
                ],
                'mitigation_strategies': [
                    'Robust model validation',
                    'Continuous monitoring and alerting',
                    'Model governance frameworks',
                    'Regular model updates and retraining'
                ]
            },
            'general': {
                'ml_workflow': [
                    'Problem definition and data collection',
                    'Exploratory data analysis',
                    'Data preprocessing and feature engineering',
                    'Model selection and training',
                    'Evaluation and validation',
                    'Deployment and monitoring'
                ],
                'common_algorithms': {
                    'supervised': ['Linear/Logistic Regression', 'Random Forest', 'XGBoost', 'Neural Networks'],
                    'unsupervised': ['K-Means', 'Hierarchical Clustering', 'PCA', 't-SNE'],
                    'reinforcement': ['Q-Learning', 'Policy Gradient', 'Actor-Critic']
                },
                'data_preprocessing': [
                    'Data quality assessment',
                    'Missing value handling',
                    'Outlier detection and treatment',
                    'Feature scaling and normalization',
                    'Feature encoding (categorical variables)',
                    'Feature selection and dimensionality reduction'
                ]
            }
        }
    
    def generate_prediction_response(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate knowledge-based response for prediction queries."""
        kb = self.knowledge_base['prediction']
        
        # Analyze query for specific prediction type
        query_lower = query.lower()
        prediction_type = 'general'
        
        if any(term in query_lower for term in ['time series', 'forecast', 'trend']):
            prediction_type = 'time_series'
        elif any(term in query_lower for term in ['classification', 'category', 'class']):
            prediction_type = 'classification'
        elif any(term in query_lower for term in ['regression', 'continuous', 'value']):
            prediction_type = 'regression'
        
        response = {
            'type': 'prediction',
            'prediction_type': prediction_type,
            'query': query,
            'recommendations': {
                'techniques': kb['techniques'],
                'key_factors': kb['factors'],
                'best_practices': kb['best_practices']
            }
        }
        
        # Add specific guidance based on prediction type
        if prediction_type == 'time_series':
            response['specific_guidance'] = {
                'recommended_models': ['ARIMA', 'SARIMA', 'Prophet', 'LSTM'],
                'considerations': [
                    'Check for seasonality and trends',
                    'Handle missing values appropriately',
                    'Consider external factors',
                    'Validate with recent data'
                ]
            }
        elif prediction_type == 'classification':
            response['specific_guidance'] = {
                'recommended_models': ['Random Forest', 'XGBoost', 'SVM', 'Neural Networks'],
                'considerations': [
                    'Handle class imbalance',
                    'Feature selection and engineering',
                    'Cross-validation strategy',
                    'Evaluation metrics selection'
                ]
            }
        elif prediction_type == 'regression':
            response['specific_guidance'] = {
                'recommended_models': ['Linear Regression', 'Random Forest', 'XGBoost', 'Neural Networks'],
                'considerations': [
                    'Check for multicollinearity',
                    'Handle outliers appropriately',
                    'Feature scaling and normalization',
                    'Residual analysis'
                ]
            }
        
        response['message'] = f"Based on your {prediction_type} prediction query, here are comprehensive recommendations and best practices."
        return response
    
    def generate_performance_response(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate knowledge-based response for performance queries."""
        kb = self.knowledge_base['performance']
        
        # Determine the type of performance evaluation
        query_lower = query.lower()
        eval_type = 'general'
        
        if any(term in query_lower for term in ['classification', 'classifier']):
            eval_type = 'classification'
        elif any(term in query_lower for term in ['regression', 'regressor']):
            eval_type = 'regression'
        elif any(term in query_lower for term in ['clustering', 'cluster']):
            eval_type = 'clustering'
        
        response = {
            'type': 'performance',
            'evaluation_type': eval_type,
            'query': query,
            'metrics': kb['metrics'].get(eval_type, kb['metrics']['classification']),
            'evaluation_methods': kb['evaluation_methods'],
            'improvement_strategies': kb['improvement_strategies']
        }
        
        # Add specific performance guidance
        if eval_type == 'classification':
            response['detailed_guidance'] = {
                'primary_metrics': ['Accuracy', 'F1-Score', 'AUC-ROC'],
                'when_to_use': {
                    'Accuracy': 'Balanced datasets with equal class importance',
                    'Precision': 'When false positives are costly',
                    'Recall': 'When false negatives are costly',
                    'F1-Score': 'Imbalanced datasets or equal importance of precision/recall'
                },
                'common_issues': ['Class imbalance', 'Overfitting', 'Data leakage']
            }
        elif eval_type == 'regression':
            response['detailed_guidance'] = {
                'primary_metrics': ['RMSE', 'MAE', 'R²'],
                'when_to_use': {
                    'RMSE': 'When large errors are particularly undesirable',
                    'MAE': 'When all errors are equally important',
                    'R²': 'Understanding proportion of variance explained',
                    'MAPE': 'When relative error is more important than absolute'
                },
                'common_issues': ['Outliers impact', 'Heteroscedasticity', 'Non-linear relationships']
            }
        
        response['message'] = f"Here's a comprehensive performance evaluation guide for {eval_type} models."
        return response
    
    def generate_risk_response(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate knowledge-based response for risk queries."""
        kb = self.knowledge_base['risk']
        
        # Identify specific risk concerns
        query_lower = query.lower()
        risk_focus = []
        
        if any(term in query_lower for term in ['model', 'overfitting', 'bias']):
            risk_focus.append('model_risk')
        if any(term in query_lower for term in ['data', 'quality', 'drift']):
            risk_focus.append('data_risk')
        if any(term in query_lower for term in ['deployment', 'production', 'operational']):
            risk_focus.append('operational_risk')
        if any(term in query_lower for term in ['compliance', 'regulatory', 'governance']):
            risk_focus.append('regulatory_risk')
        
        if not risk_focus:
            risk_focus = ['general']
        
        response = {
            'type': 'risk',
            'risk_focus': risk_focus,
            'query': query,
            'risk_types': kb['types'],
            'assessment_methods': kb['assessment_methods'],
            'mitigation_strategies': kb['mitigation_strategies']
        }
        
        # Add specific risk guidance
        risk_details = {}
        if 'model_risk' in risk_focus:
            risk_details['model_risk'] = {
                'key_indicators': ['High variance in predictions', 'Poor generalization', 'Unstable feature importance'],
                'mitigation': ['Robust validation', 'Regularization', 'Ensemble methods', 'Feature selection']
            }
        if 'data_risk' in risk_focus:
            risk_details['data_risk'] = {
                'key_indicators': ['Data drift', 'Missing values', 'Outliers', 'Inconsistent formats'],
                'mitigation': ['Data monitoring', 'Quality checks', 'Robust preprocessing', 'Anomaly detection']
            }
        if 'operational_risk' in risk_focus:
            risk_details['operational_risk'] = {
                'key_indicators': ['Performance degradation', 'System failures', 'Scalability issues'],
                'mitigation': ['Monitoring systems', 'A/B testing', 'Gradual rollouts', 'Fallback mechanisms']
            }
        
        response['detailed_guidance'] = risk_details
        response['message'] = f"Comprehensive risk assessment for: {', '.join(risk_focus)}"
        return response
    
    def generate_general_response(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate knowledge-based response for general queries."""
        kb = self.knowledge_base['general']
        
        # Analyze query for ML/Data Science topics
        query_lower = query.lower()
        
        topics = []
        if any(term in query_lower for term in ['machine learning', 'ml', 'algorithm']):
            topics.append('machine_learning')
        if any(term in query_lower for term in ['data science', 'analytics', 'analysis']):
            topics.append('data_science')
        if any(term in query_lower for term in ['feature', 'preprocessing', 'cleaning']):
            topics.append('data_preprocessing')
        if any(term in query_lower for term in ['model selection', 'comparison', 'choose']):
            topics.append('model_selection')
        
        # If no specific topics found, provide general guidance
        if not topics:
            topics = ['general_guidance']
        
        response = {
            'type': 'general',
            'topics': topics,
            'query': query,
            'workflow': kb['ml_workflow'],
            'algorithms': kb['common_algorithms'],
            'preprocessing': kb['data_preprocessing']
        }
        
        # Provide topic-specific guidance
        if 'machine_learning' in topics:
            response['ml_guidance'] = {
                'workflow': kb['ml_workflow'],
                'common_algorithms': kb['common_algorithms']
            }
        
        if 'data_preprocessing' in topics:
            response['preprocessing_guidance'] = {
                'steps': kb['data_preprocessing'],
                'techniques': {
                    'missing_values': ['Imputation', 'Deletion', 'Model-based methods'],
                    'scaling': ['StandardScaler', 'MinMaxScaler', 'RobustScaler'],
                    'encoding': ['One-hot encoding', 'Label encoding', 'Target encoding']
                }
            }
        
        if 'general_guidance' in topics:
            response['general_guidance'] = {
                'message': 'I can help with machine learning, data science, and analytics questions.',
                'areas_of_expertise': [
                    'Model prediction and forecasting',
                    'Performance evaluation and metrics',
                    'Risk assessment and mitigation',
                    'Data preprocessing and feature engineering',
                    'Model selection and comparison'
                ]
            }
        
        response['message'] = "Here's comprehensive guidance based on your query."
        return response


class QueryLogger:
    """Handles logging of queries to separate files."""
    
    def __init__(self, log_folder: str = "logs"):
        self.log_folder = log_folder
        os.makedirs(self.log_folder, exist_ok=True)
        self.raw_queries_file = os.path.join(self.log_folder, "raw_queries.jsonl")
        self.answered_queries_file = os.path.join(self.log_folder, "answered_queries.jsonl")
    
    def log_raw_query(self, query_id: str, user_id: str, query: str, domain: str = None):
        """Log raw query to raw_queries.jsonl file."""
        timestamp = datetime.now()
        raw_entry = {
            "query_id": query_id,
            "user_id": user_id,
            "query": query,
            "domain": domain,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime('%Y-%m-%d'),
            "time": timestamp.strftime('%H:%M:%S')
        }
        
        try:
            with open(self.raw_queries_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(raw_entry, ensure_ascii=False, default=str) + '\n')
        except Exception as e:
            print(f"Error logging raw query: {e}")
    
    def log_answered_query(self, query_id: str, user_id: str, query: str, response: Dict[str, Any], domain: str = None):
        """Log query with response to answered_queries.jsonl file."""
        timestamp = datetime.now()
        answered_entry = {
            "query_id": query_id,
            "user_id": user_id,
            "query": query,
            "response": response,
            "domain": domain,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime('%Y-%m-%d'),
            "time": timestamp.strftime('%H:%M:%S'),
            "response_type": response.get('type', 'unknown'),
            "intent": response.get('intent', 'unknown'),
            "confidence": response.get('confidence', 0.0),
            "has_error": response.get('type') == 'error' or 'error' in response
        }
        
        try:
            with open(self.answered_queries_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(answered_entry, ensure_ascii=False, default=str) + '\n')
        except Exception as e:
            print(f"Error logging answered query: {e}")


class QueryHistory:
    """Manages query history for users."""
    
    def __init__(self, storage_file: str = "query_history.json"):
        self.storage_file = storage_file
        self.history = self._load_history()
    
    def _load_history(self) -> Dict[str, List[Dict]]:
        """Load query history from storage file."""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load query history: {e}")
            return {}
    
    def _save_history(self):
        """Save query history to storage file."""
        try:
            os.makedirs(os.path.dirname(self.storage_file) if os.path.dirname(self.storage_file) else '.', exist_ok=True)
            
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"Error saving history: {e}")
    
    def add_query(self, user_id: str, query: str, response: Dict[str, Any], 
                  domain: str = None) -> str:
        """Add a query to history."""
        timestamp = datetime.now()
        query_id = hashlib.md5(f"{user_id}_{timestamp.isoformat()}".encode()).hexdigest()[:8]
        
        title = self._generate_title(query)
        
        query_record = {
            'query_id': query_id,
            'title': title,
            'query': query,
            'response': response,
            'domain': domain,
            'timestamp': timestamp.isoformat(),
            'date': timestamp.strftime('%Y-%m-%d'),
            'time': timestamp.strftime('%H:%M:%S')
        }
        
        if user_id not in self.history:
            self.history[user_id] = []
        
        self.history[user_id].append(query_record)
        self._save_history()
        
        return query_id
    
    def _generate_title(self, query: str) -> str:
        """Generate a meaningful title from the query."""
        clean_query = ' '.join(query.split())
        
        sentences = clean_query.split('.')
        if len(sentences) > 1 and len(sentences[0]) < 60:
            return sentences[0].strip() + '.'
        
        if len(clean_query) <= 50:
            return clean_query
        
        words = clean_query[:50].split()
        if len(words) > 1:
            return ' '.join(words[:-1]) + '...'
        return clean_query[:50] + '...' if len(clean_query) > 50 else clean_query
    
    def get_user_history(self, user_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get query history for a user."""
        user_history = self.history.get(user_id, [])
        sorted_history = sorted(user_history, key=lambda x: x['timestamp'], reverse=True)
        
        if limit:
            return sorted_history[:limit]
        return sorted_history
    
    def get_query_by_id(self, user_id: str, query_id: str) -> Optional[Dict]:
        """Get a specific query by ID."""
        user_history = self.history.get(user_id, [])
        for query_record in user_history:
            if query_record['query_id'] == query_id:
                return query_record
        return None
    
    def search_history(self, user_id: str, search_term: str) -> List[Dict]:
        """Search through user's query history."""
        user_history = self.history.get(user_id, [])
        results = []
        
        search_term_lower = search_term.lower()
        for query_record in user_history:
            query_text = query_record.get('query', '').lower()
            title_text = query_record.get('title', '').lower()
            
            if (search_term_lower in query_text or search_term_lower in title_text):
                results.append(query_record)
        
        return sorted(results, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_query(self, user_id: str, query_id: str) -> bool:
        """Delete a specific query from history."""
        if user_id in self.history:
            original_length = len(self.history[user_id])
            self.history[user_id] = [
                q for q in self.history[user_id] 
                if q.get('query_id') != query_id
            ]
            if len(self.history[user_id]) < original_length:
                self._save_history()
                return True
        return False
    
    def clear_user_history(self, user_id: str) -> bool:
        """Clear all history for a user."""
        if user_id in self.history:
            self.history[user_id] = []
            self._save_history()
            return True
        return False
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about user's query history."""
        user_history = self.history.get(user_id, [])
        
        if not user_history:
            return {
                'total_queries': 0,
                'domains': {},
                'date_range': None
            }
        
        domain_counts = {}
        dates = []
        
        for query in user_history:
            domain = query.get('domain', 'unknown')
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            dates.append(query.get('date'))
        
        valid_dates = [d for d in dates if d]
        date_range = None
        if valid_dates:
            valid_dates.sort()
            date_range = {
                'earliest': valid_dates[0],
                'latest': valid_dates[-1]
            }
        
        return {
            'total_queries': len(user_history),
            'domains': domain_counts,
            'date_range': date_range
        }


class QueryHandler:
    """Enhanced query handler with knowledge-based responses and comprehensive logging."""
    
    INTENT_HANDLERS = {
        'prediction': 'handle_prediction_query',
        'performance': 'handle_performance_query',
        'risk': 'handle_risk_query',
        'general': 'handle_general_query',
        'history': 'handle_history_query'
    }
    
    INTENT_KEYWORDS = {
        'prediction': ['predict', 'forecast', 'future', 'estimate', 'projection', 'next', 'upcoming'],
        'performance': ['performance', 'accuracy', 'metrics', 'evaluate', 'score', 'results'],
        'risk': ['risk', 'danger', 'threat', 'hazard', 'vulnerability', 'exposure'],
        'history': ['history', 'past', 'previous', 'show', 'list', 'search', 'clear']
    }
    
    def __init__(self, logger: Logger = None, llm_manager: Optional[LLMManager] = None):
        self.logger = logger or self._create_default_logger()
        self.llm_manager = llm_manager
        self.query_history = QueryHistory()
        self.knowledge_handler = KnowledgeBaseHandler(self.logger)
        self.query_logger = QueryLogger()
        
    def _create_default_logger(self) -> Logger:
        """Create a default logger if none provided."""
        try:
            return Logger()
        except:
            class SimpleLogger:
                def log_info(self, msg): print(f"INFO: {msg}")
                def log_warning(self, msg): print(f"WARNING: {msg}")
                def log_error(self, msg): print(f"ERROR: {msg}")
            return SimpleLogger()
        
    def classify_query(self, query: str, max_retries: int = 2) -> Dict[str, Any]:
        """Classify query intent using LLM with fallback to keyword-based classification."""
        if self._is_history_query(query):
            return {'intent': 'history', 'confidence': 1.0}
        
        if self.llm_manager:
            for attempt in range(max_retries + 1):
                try:
                    query_result = self.llm_manager.process_query([query])[0]
                    intent = query_result.get('intent', 'general')
                    confidence = query_result.get('confidence', 0.5)
                    
                    if intent not in self.INTENT_HANDLERS:
                        self.logger.log_warning(f"Unknown intent '{intent}', falling back to keyword classification")
                        return self._classify_by_keywords(query)
                    
                    self.logger.log_info(f"LLM classified query intent: {intent} (confidence: {confidence})")
                    return {'intent': intent, 'confidence': confidence}
                    
                except Exception as e:
                    self.logger.log_warning(f"LLM classification attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries:
                        self.logger.log_warning("All LLM classification attempts failed, falling back to keyword classification")
                        return self._classify_by_keywords(query)
        else:
            self.logger.log_info("No LLM manager available, using keyword classification")
        
        return self._classify_by_keywords(query)
    
    def _classify_by_keywords(self, query: str) -> Dict[str, Any]:
        """Classify query intent based on keywords."""
        query_lower = query.lower()
        scores = {}
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[intent] = score
        
        if scores:
            best_intent = max(scores, key=scores.get)
            confidence = min(scores[best_intent] * 0.3, 0.9)
            self.logger.log_info(f"Keyword classified query intent: {best_intent} (confidence: {confidence})")
            return {'intent': best_intent, 'confidence': confidence}
        else:
            self.logger.log_info("No keywords matched, defaulting to general intent")
            return {'intent': 'general', 'confidence': 0.5}
    
    def _is_history_query(self, query: str) -> bool:
        """Check if query is asking for history/past queries."""
        history_keywords = [
            'history', 'past queries', 'previous questions', 'show my queries',
            'what did i ask', 'my questions', 'search history', 'past questions',
            'show history', 'query history', 'recent queries', 'last queries'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in history_keywords)

    def _generate_user_id(self) -> str:
        """Generate a default user ID if none provided."""
        return f"user_{str(uuid.uuid4())[:8]}"

    def handle_query(self, query: str, domain_config: Any = None, raw_data: pd.DataFrame = None,
                 processed_data: Dict[str, Any] = None, models: Dict[str, Any] = None, 
                 user_id: str = None, domain: str = None, model_handler: Any = None) -> Dict[str, Any]:
        """Handle query with knowledge-based responses and comprehensive logging."""
        
        if user_id is None:
            user_id = self._generate_user_id()
            self.logger.log_info(f"Generated user_id: {user_id}")
        
        # Generate query ID first
        timestamp = datetime.now()
        query_id = hashlib.md5(f"{user_id}_{timestamp.isoformat()}_{query}".encode()).hexdigest()[:8]
        
        # Log raw query immediately
        self.query_logger.log_raw_query(query_id, user_id, query, domain)
        
        intent = 'unknown'
        confidence = 0.0
        
        try:
            classification_result = self.classify_query(query)
            intent = classification_result['intent']
            confidence = classification_result['confidence']
            
            if confidence < 0.7:
                self.logger.log_warning(f"Low confidence classification: {intent} ({confidence})")
            
            response = None
            
            # Step 1: Try model-specific Q&A if model_handler is available
            if model_handler is not None and intent not in ['history']:
                try:
                    model_answer = model_handler.answer_question(question=query, domain=domain)
                    if model_answer.get("has_answer"):
                        self.logger.log_info("Answered using ModelHandler.")
                        model_answer.update({
                            'intent': 'model_specific',
                            'confidence': 1.0,
                            'user_id': user_id,
                            'query_id': query_id
                        })
                        
                        # Log the answered query
                        self.query_logger.log_answered_query(query_id, user_id, query, model_answer, domain)
                        
                        # Save to history
                        self.query_history.add_query(user_id, query, model_answer, domain)
                        
                        return model_answer
                except Exception as e:
                    self.logger.log_warning(f"ModelHandler answer failed: {str(e)}")

            # Handle history queries directly
            if intent == 'history':
                result = self._handle_history_query(query, user_id)
                response = {
                    'result': result,
                    'intent': intent,
                    'confidence': confidence,
                    'query': query,
                    'user_id': user_id,
                    'query_id': query_id,
                    'is_history': True,
                    'type': 'history',
                    'message': result.get('message', 'History query processed')
                }
            else:
                # Generate knowledge-based response
                context = {
                    'raw_data': raw_data,
                    'processed_data': processed_data,
                    'models': models,
                    'domain': domain
                }
                
                if intent == 'prediction':
                    response = self.knowledge_handler.generate_prediction_response(query, context)
                elif intent == 'performance':
                    response = self.knowledge_handler.generate_performance_response(query,