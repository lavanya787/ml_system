import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration
import torch
from typing import Dict, List, Any
from logger import Logger

class LLMManager:
    def __init__(self, logger: Logger):
        self.logger = logger
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.intent_model_name = 'distilbert-base-uncased'
        self.response_model_name = 't5-small'
        self.intent_tokenizer = None
        self.intent_model = None
        self.response_tokenizer = None
        self.response_model = None
        self._load_models()
    
    def _load_models(self) -> None:
        """Load DistilBERT and T5 models."""
        try:
            # Load DistilBERT for intent classification
            self.intent_tokenizer = DistilBertTokenizer.from_pretrained(self.intent_model_name)
            self.intent_model = DistilBertForSequenceClassification.from_pretrained(self.intent_model_name).to(self.device)
            self.intent_model.eval()
            self.logger.log_info(f"Loaded DistilBERT model on {self.device}")
            
            # Load T5 for response generation
            self.response_tokenizer = T5Tokenizer.from_pretrained(self.response_model_name)
            self.response_model = T5ForConditionalGeneration.from_pretrained(self.response_model_name).to(self.device)
            self.logger.log_info(f"Loaded T5 model on {self.device}")
        except Exception as e:
            self.logger.log_error(f"Failed to load models: {str(e)}")
            raise
    
    def process_query(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Classify query intent using DistilBERT."""
        try:
            inputs = self.intent_tokenizer(queries, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.intent_model(**inputs)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=-1)
            predicted_classes = torch.argmax(predictions, dim=1)
            
            # Map class IDs to intent labels (example mapping)
            intent_map = {0: 'prediction', 1: 'performance', 2: 'risk', 3: 'general'}
            results = []
            for idx, query in enumerate(queries):
                intent_id = predicted_classes[idx].item()
                results.append({
                    'query': query,
                    'intent': intent_map.get(intent_id, 'general'),
                    'confidence': predictions[idx][intent_id].item()
                })
            self.logger.log_info(f"Processed {len(queries)} queries for intent classification")
            return results
        except Exception as e:
            self.logger.log_error(f"Query processing failed: {str(e)}")
            return [{'query': q, 'intent': 'general', 'confidence': 0.0} for q in queries]
    
    def generate_response(self, context: str) -> str:
        """Generate a response using T5."""
        try:
            input_text = f"Generate a response for: {context}"
            inputs = self.response_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            outputs = self.response_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,
                num_beams=5,
                early_stopping=True
            )
            response = self.response_tokenizer.decode(outputs[0], skip_special_tokens=True)
            self.logger.log_info("Generated T5 response")
            return response
        except Exception as e:
            self.logger.log_error(f"Response generation failed: {str(e)}")
            return "Unable to generate response due to an error."