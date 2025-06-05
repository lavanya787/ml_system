import tensorflow as tf
import numpy as np
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
from transformers import TFT5ForConditionalGeneration, T5Tokenizer
from typing import Dict, Any, List
import streamlit as st
from cachetools import LRUCache
import hashlib

class LLMManager:
    def __init__(self, distilbert_model: str = "distilbert-base-uncased", 
                 t5_model: str = "t5-small", num_labels: int = 26, cache_size: int = 1000):
        """Initialize LLM for domain classification, query processing, and response generation"""
        # DistilBERT for domain/intent classification
        self.distilbert_model_name = distilbert_model
        self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained(distilbert_model)
        try:
            self.distilbert_model = TFDistilBertForSequenceClassification.from_pretrained(
                distilbert_model, num_labels=num_labels
            )
        except Exception as e:
            st.warning(f"DistilBERT loading failed: {str(e)}. Initializing new model.")
            self.distilbert_model = TFDistilBertForSequenceClassification.from_pretrained(
                distilbert_model, num_labels=num_labels, from_tf=True
            )
        
        # T5 for response generation
        self.t5_model_name = t5_model
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_model)
        try:
            self.t5_model = TFT5ForConditionalGeneration.from_pretrained(t5_model)
        except Exception as e:
            st.warning(f"T5 loading failed: {str(e)}. Initializing new model.")
            self.t5_model = TFT5ForConditionalGeneration.from_pretrained(t5_model)
        
        self.domains = [
            'healthcare', 'ecommerce', 'finance', 'education', 'travel', 'customer_support',
            'entertainment', 'gaming', 'legal', 'marketing', 'logistics', 'real_estate',
            'manufacturing', 'agriculture', 'energy', 'hospitality', 'automobile',
            'telecommunications', 'government', 'food_beverage', 'it_services',
            'event_management', 'insurance', 'retail', 'hr_resources', 'banking'
        ]
        self.domain_to_idx = {domain: idx for idx, domain in enumerate(self.domains)}
        
        # Initialize cache
        self.prediction_cache = LRUCache(maxsize=cache_size)
        self.response_cache = LRUCache(maxsize=cache_size)
    
    def _hash_input(self, text: str) -> str:
        """Create a hash for caching"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def preprocess_text(self, texts: List[str], max_length: int = 128, model_type: str = 'distilbert') -> Dict[str, Any]:
        """Tokenize input texts for DistilBERT or T5"""
        tokenizer = self.distilbert_tokenizer if model_type == 'distilbert' else self.t5_tokenizer
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )
        return encodings
    
    def predict_domain(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict domains for a batch of text inputs"""
        results = []
        texts_to_process = []
        cached_results = []
        indices_to_process = []
        
        # Check cache
        for i, text in enumerate(texts):
            cache_key = self._hash_input(f"domain:{text}")
            if cache_key in self.prediction_cache:
                cached_results.append((i, self.prediction_cache[cache_key]))
            else:
                texts_to_process.append(text)
                indices_to_process.append(i)
        
        # Process uncached texts
        if texts_to_process:
            encodings = self.preprocess_text(texts_to_process, model_type='distilbert')
            outputs = self.distilbert_model(encodings)
            logits = outputs.logits
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()
            
            for i, prob in enumerate(probabilities):
                predicted_idx = np.argmax(prob)
                predicted_domain = self.domains[predicted_idx]
                confidence = prob[predicted_idx]
                result = {
                    'domain': predicted_domain,
                    'confidence': float(confidence),
                    'probabilities': {domain: float(p) for domain, p in zip(self.domains, prob)}
                }
                cache_key = self._hash_input(f"domain:{texts_to_process[i]}")
                self.prediction_cache[cache_key] = result
                results.append((indices_to_process[i], result))
        
        # Combine cached and new results
        results.extend(cached_results)
        results = sorted(results, key=lambda x: x[0])  # Sort by original index
        return [r[1] for r in results]
    
    def process_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of natural language queries"""
        results = []
        queries_to_process = []
        cached_results = []
        indices_to_process = []
        
        # Check cache
        for i, query in enumerate(queries):
            cache_key = self._hash_input(f"query:{query}")
            if cache_key in self.prediction_cache:
                cached_results.append((i, self.prediction_cache[cache_key]))
            else:
                queries_to_process.append(query)
                indices_to_process.append(i)
        
        # Process uncached queries
        if queries_to_process:
            encodings = self.preprocess_text(queries_to_process, model_type='distilbert')
            outputs = self.distilbert_model(encodings)
            logits = outputs.logits
            probabilities = tf.nn.softmax(logits, axis=-1).numpy()
            
            for i, prob in enumerate(probabilities):
                predicted_idx = np.argmax(prob)
                intent = self._map_query_to_intent(queries_to_process[i])
                result = {
                    'intent': intent,
                    'confidence': float(prob[predicted_idx]),
                    'raw_query': queries_to_process[i]
                }
                cache_key = self._hash_input(f"query:{queries_to_process[i]}")
                self.prediction_cache[cache_key] = result
                results.append((indices_to_process[i], result))
        
        # Combine cached and new results
        results.extend(cached_results)
        results = sorted(results, key=lambda x: x[0])
        return [r[1] for r in results]
    
    def generate_response(self, context: str, max_length: int = 100) -> str:
        """Generate a natural language response using T5"""
        cache_key = self._hash_input(f"response:{context}")
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        input_text = f"summarize: {context}"
        encodings = self.preprocess_text([input_text], max_length=512, model_type='t5')
        generated_ids = self.t5_model.generate(
            input_ids=encodings['input_ids'],
            attention_mask=encodings['attention_mask'],
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        response = self.t5_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        self.response_cache[cache_key] = response
        return response
    
    def _map_query_to_intent(self, query: str) -> str:
        """Map query to intent based on keywords"""
        query_lower = query.lower()
        if any(word in query_lower for word in ['predict', 'forecast']):
            return 'prediction'
        elif any(word in query_lower for word in ['correlation', 'relationship']):
            return 'correlation'
        elif any(word in query_lower for word in ['compare', 'versus']):
            return 'comparison'
        elif any(word in query_lower for word in ['average', 'mean', 'statistics']):
            return 'statistics'
        elif any(word in query_lower for word in ['performance', 'sales', 'grade']):
            return 'performance'
        elif any(word in query_lower for word in ['risk', 'churn', 'failure']):
            return 'risk'
        elif any(word in query_lower for word in ['factor', 'affect', 'influence']):
            return 'factors'
        elif any(word in query_lower for word in ['distribution', 'spread']):
            return 'distribution'
        return 'general'