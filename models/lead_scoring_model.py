import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
import asyncio
import httpx
from groq import Groq
import time

from config import get_settings
from features.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)

class LeadScoringModel:
    """Lead scoring model using XGBoost + LLM re-ranker"""
    
    def __init__(self):
        self.settings = get_settings()
        self.feature_engineer = FeatureEngineering()
        self.scaler = StandardScaler()
        self.xgb_model = None
        self.groq_client = Groq(api_key=self.settings.groq_api_key) if self.settings.groq_api_key else None
        self.feature_names = self.feature_engineer.get_feature_names()
        self.is_trained = False
        
    def train(self, training_data: pd.DataFrame, target_column: str = 'converted') -> Dict[str, Any]:
        """Train the XGBoost model"""
        logger.info("Starting model training...")
        
        try:
            # Prepare features
            X = self._prepare_features(training_data)
            y = training_data[target_column].astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train XGBoost model
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            
            self.xgb_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                verbose=False
            )
            
            # Evaluate model
            train_score = self.xgb_model.score(X_train_scaled, y_train)
            test_score = self.xgb_model.score(X_test_scaled, y_test)
            
            self.is_trained = True
            
            training_results = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'feature_importance': dict(zip(
                    self.feature_names, 
                    self.xgb_model.feature_importances_
                )),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Model training completed. Test accuracy: {test_score:.4f}")
            return training_results
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            raise
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from raw data"""
        features_list = []
        
        for _, row in data.iterrows():
            # Convert row to lead data format
            lead_data = row.to_dict()
            
            # Extract features
            features = self.feature_engineer.extract_features(lead_data)
            
            # Ensure all expected features are present
            feature_vector = {}
            for feature_name in self.feature_names:
                feature_vector[feature_name] = features.get(feature_name, 0.0)
            
            features_list.append(feature_vector)
        
        return pd.DataFrame(features_list)
    
    async def predict_single(self, lead_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict score for a single lead with LLM re-ranking"""
        start_time = time.time()
        
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        try:
            # Extract features
            features = self.feature_engineer.extract_features(lead_data)
            
            # Prepare feature vector
            feature_vector = []
            for feature_name in self.feature_names:
                feature_vector.append(features.get(feature_name, 0.0))
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Get base prediction from XGBoost
            base_proba = self.xgb_model.predict_proba(feature_vector_scaled)[0]
            base_score = base_proba[1]  # Probability of positive class
            
            # LLM re-ranking
            llm_result = await self._llm_rerank(lead_data, base_score, features)
            llm_adjusted_score = llm_result['adjusted_score']
            reasoning = llm_result['reasoning']
            
            # Calculate final score (0-100 scale)
            final_score = llm_adjusted_score * 100
            
            # Determine priority
            priority = self._determine_priority(final_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(features, final_score)
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                'lead_id': lead_data.get('lead_id', 'unknown'),
                'base_score': base_score,
                'llm_adjusted_score': llm_adjusted_score,
                'final_score': final_score,
                'confidence': max(base_proba),  # Confidence from XGB
                'reasoning': reasoning,
                'priority': priority,
                'recommendations': recommendations,
                'processing_time_ms': processing_time,
                'feature_scores': self._get_top_features(features),
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for lead {lead_data.get('lead_id', 'unknown')}: {str(e)}")
            raise
    
    async def predict_batch(self, leads_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict scores for multiple leads"""
        start_time = time.time()
        
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        try:
            # Prepare all features
            all_features = []
            for lead_data in leads_data:
                features = self.feature_engineer.extract_features(lead_data)
                feature_vector = []
                for feature_name in self.feature_names:
                    feature_vector.append(features.get(feature_name, 0.0))
                all_features.append((lead_data, features, feature_vector))
            
            # Batch prediction with XGBoost
            feature_matrix = [fv for _, _, fv in all_features]
            feature_matrix_scaled = self.scaler.transform(feature_matrix)
            base_probas = self.xgb_model.predict_proba(feature_matrix_scaled)
            
            # Process each lead with LLM re-ranking
            results = []
            for i, (lead_data, features, _) in enumerate(all_features):
                base_score = base_probas[i][1]
                
                # LLM re-ranking for high-scoring leads only (optimization)
                if base_score > 0.5:
                    llm_result = await self._llm_rerank(lead_data, base_score, features)
                    llm_adjusted_score = llm_result['adjusted_score']
                    reasoning = llm_result['reasoning']
                else:
                    llm_adjusted_score = base_score
                    reasoning = "Standard scoring (below LLM threshold)"
                
                final_score = llm_adjusted_score * 100
                priority = self._determine_priority(final_score)
                recommendations = self._generate_recommendations(features, final_score)
                
                results.append({
                    'lead_id': lead_data.get('lead_id', f'unknown_{i}'),
                    'base_score': base_score,
                    'llm_adjusted_score': llm_adjusted_score,
                    'final_score': final_score,
                    'confidence': max(base_probas[i]),
                    'reasoning': reasoning,
                    'priority': priority,
                    'recommendations': recommendations,
                    'feature_scores': self._get_top_features(features),
                    'timestamp': datetime.utcnow().isoformat()
                })
            
            processing_time = (time.time() - start_time) * 1000
            
            # Add batch metadata
            for result in results:
                result['batch_processing_time_ms'] = processing_time
                result['batch_size'] = len(leads_data)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise
    
    async def _llm_rerank(self, lead_data: Dict[str, Any], base_score: float, features: Dict[str, float]) -> Dict[str, Any]:
        """Use LLM to re-rank the lead score based on context"""
        if not self.groq_client:
            return {'adjusted_score': base_score, 'reasoning': 'LLM not available'}
        
        try:
            # Prepare context for LLM
            context = self._prepare_llm_context(lead_data, base_score, features)
            
            prompt = f"""
You are an expert real estate lead scoring analyst. Analyze the following lead data and adjust the base ML model score.

LEAD CONTEXT:
{context}

BASE ML SCORE: {base_score:.3f} (0-1 scale)

TASK:
1. Analyze the lead's behavior, demographics, and market context
2. Identify key signals that might adjust the score up or down
3. Provide an adjusted score (0-1 scale) and clear reasoning

Consider these factors:
- Recent behavioral patterns indicating urgency
- Financial readiness signals
- Market timing factors
- Communication patterns
- Search specificity and intent

Respond in JSON format:
{{
    "adjusted_score": 0.XXX,
    "reasoning": "Brief explanation of key factors influencing the adjustment",
    "key_signals": ["signal1", "signal2", "signal3"]
}}
"""
            
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model=self.settings.groq_model,
                messages=[
                    {"role": "system", "content": "You are an expert real estate lead scoring analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                llm_result = json.loads(response_text)
                adjusted_score = max(0.0, min(1.0, float(llm_result.get('adjusted_score', base_score))))
                reasoning = llm_result.get('reasoning', 'LLM adjustment applied')
            except json.JSONDecodeError:
                # Fallback if LLM doesn't return valid JSON
                adjusted_score = base_score
                reasoning = f"LLM response parsing failed. Using base score. Response: {response_text[:100]}"
            
            return {
                'adjusted_score': adjusted_score,
                'reasoning': reasoning
            }
            
        except Exception as e:
            logger.warning(f"LLM re-ranking failed: {str(e)}")
            return {'adjusted_score': base_score, 'reasoning': f'LLM error: {str(e)}'}
    
    def _prepare_llm_context(self, lead_data: Dict[str, Any], base_score: float, features: Dict[str, float]) -> str:
        """Prepare context string for LLM analysis"""
        context_parts = []
        
        # Behavioral summary
        behavioral = lead_data.get('behavioral', {})
        context_parts.append(f"BEHAVIORAL: {behavioral.get('property_interaction_frequency', 0)} interactions, "
                           f"{behavioral.get('search_frequency', 0)} searches, "
                           f"{behavioral.get('contact_attempts', 0)} contact attempts")
        
        # Demographic summary
        demographic = lead_data.get('demographic', {})
        context_parts.append(f"DEMOGRAPHIC: {demographic.get('income_bracket', 'unknown')} income, "
                           f"{demographic.get('family_composition', 'unknown')} family, "
                           f"{demographic.get('age_range', 'unknown')} age")
        
        # Key feature scores
        top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:5]
        feature_summary = ", ".join([f"{k}: {v:.2f}" for k, v in top_features])
        context_parts.append(f"TOP FEATURES: {feature_summary}")
        
        # Third-party signals
        third_party = lead_data.get('third_party', {})
        if third_party.get('credit_inquiry_activity'):
            context_parts.append("FINANCIAL: Recent credit inquiries detected")
        
        return " | ".join(context_parts)
    
    def _determine_priority(self, final_score: float) -> str:
        """Determine lead priority based on final score"""
        if final_score >= 75:
            return "high"
        elif final_score >= 50:
            return "medium"
        else:
            return "low"
    
    def _generate_recommendations(self, features: Dict[str, float], final_score: float) -> List[str]:
        """Generate action recommendations based on features and score"""
        recommendations = []
        
        if final_score >= 75:
            recommendations.append("URGENT: Contact within 2 hours")
            recommendations.append("Offer premium properties first")
            
        if features.get('contact_attempts', 0) > 2:
            recommendations.append("Lead is actively seeking contact - prioritize immediate response")
            
        if features.get('credit_inquiry_activity', 0) > 0:
            recommendations.append("Financial readiness confirmed - discuss financing options")
            
        if features.get('search_query_specificity', 0) > 0.7:
            recommendations.append("Specific requirements identified - provide targeted listings")
            
        if features.get('engagement_score', 0) > 0.8:
            recommendations.append("Highly engaged - schedule site visit")
            
        if features.get('income_bracket_encoded', 0) >= 3:
            recommendations.append("High-value prospect - assign senior agent")
            
        if not recommendations:
            recommendations.append("Standard follow-up process")
            
        return recommendations[:4]  # Limit to top 4 recommendations
    
    def _get_top_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get top contributing features for explainability"""
        sorted_features = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_features[:10])  # Top 10 features
    
    def save_model(self, filepath: str):
        """Save the trained model and preprocessing components"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'xgb_model': self.xgb_model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        try:
            model_data = joblib.load(filepath)
            self.xgb_model = model_data['xgb_model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Failed to load model from {filepath}: {str(e)}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        
        # Convert numpy.float32 to Python float for JSON serialization
        importance_values = [float(val) for val in self.xgb_model.feature_importances_]
        return dict(zip(self.feature_names, importance_values))
