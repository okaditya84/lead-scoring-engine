import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Feature engineering pipeline for lead scoring"""
    
    def __init__(self):
        self.feature_specs = self._initialize_feature_specs()
    
    def _initialize_feature_specs(self) -> Dict[str, Any]:
        """Initialize feature specifications and transformations"""
        return {
            'behavioral': {
                'interaction_features': [
                    'property_interaction_frequency',
                    'search_query_specificity', 
                    'time_spent_on_platform',
                    'property_views_count',
                    'search_frequency',
                    'page_depth',
                    'return_visits',
                    'contact_attempts'
                ],
                'derived_features': [
                    'engagement_score',
                    'intent_velocity',
                    'platform_stickiness'
                ]
            },
            'demographic': {
                'categorical_features': [
                    'income_bracket',
                    'family_composition', 
                    'age_range',
                    'occupation'
                ],
                'derived_features': [
                    'buying_power_score',
                    'life_stage_score'
                ]
            }
        }
    
    def extract_features(self, lead_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract and engineer features from lead data"""
        features = {}
        
        # Extract behavioral features
        behavioral_features = self._extract_behavioral_features(lead_data.get('behavioral', {}))
        features.update(behavioral_features)
        
        # Extract demographic features
        demographic_features = self._extract_demographic_features(lead_data.get('demographic', {}))
        features.update(demographic_features)
        
        # Extract public data features
        public_features = self._extract_public_data_features(lead_data.get('public_data', {}))
        features.update(public_features)
        
        # Extract third-party features
        third_party_features = self._extract_third_party_features(lead_data.get('third_party', {}))
        features.update(third_party_features)
        
        # Create interaction features
        interaction_features = self._create_interaction_features(features)
        features.update(interaction_features)
        
        # Add temporal features
        temporal_features = self._create_temporal_features(lead_data)
        features.update(temporal_features)
        
        return features
    
    def _extract_behavioral_features(self, behavioral_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract and engineer behavioral features"""
        features = {}
        
        # Direct behavioral features
        features['property_interaction_frequency'] = float(behavioral_data.get('property_interaction_frequency', 0))
        features['search_query_specificity'] = float(behavioral_data.get('search_query_specificity', 0))
        features['time_spent_minutes'] = float(behavioral_data.get('time_spent_on_platform', 0))
        features['property_views_count'] = float(behavioral_data.get('property_views_count', 0))
        features['search_frequency'] = float(behavioral_data.get('search_frequency', 0))
        features['page_depth'] = float(behavioral_data.get('page_depth', 0))
        features['return_visits'] = float(behavioral_data.get('return_visits', 0))
        features['contact_attempts'] = float(behavioral_data.get('contact_attempts', 0))
        
        # Derived behavioral features
        features['engagement_score'] = self._calculate_engagement_score(behavioral_data)
        features['intent_velocity'] = self._calculate_intent_velocity(behavioral_data)
        features['platform_stickiness'] = self._calculate_platform_stickiness(behavioral_data)
        features['search_intensity'] = self._calculate_search_intensity(behavioral_data)
        features['interaction_depth'] = self._calculate_interaction_depth(behavioral_data)
        
        return features
    
    def _extract_demographic_features(self, demographic_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract and engineer demographic features"""
        features = {}
        
        # Income bracket encoding
        income_mapping = {'low': 1, 'medium': 2, 'high': 3, 'premium': 4}
        features['income_bracket_encoded'] = float(income_mapping.get(
            demographic_data.get('income_bracket', '').lower(), 0
        ))
        
        # Family composition encoding
        family_mapping = {
            'single': 1, 'couple': 2, 
            'family_with_children': 3, 'joint_family': 4
        }
        features['family_composition_encoded'] = float(family_mapping.get(
            demographic_data.get('family_composition', '').lower(), 0
        ))
        
        # Age range encoding
        age_mapping = {'18-25': 1, '26-35': 2, '36-45': 3, '46-60': 4, '60+': 5}
        features['age_range_encoded'] = float(age_mapping.get(
            demographic_data.get('age_range', ''), 0
        ))
        
        # Derived demographic features
        features['buying_power_score'] = self._calculate_buying_power_score(demographic_data)
        features['life_stage_score'] = self._calculate_life_stage_score(demographic_data)
        features['area_preference_diversity'] = len(demographic_data.get('preferred_areas', []))
        
        return features
    
    def _extract_public_data_features(self, public_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract public data features"""
        features = {}
        
        price_trends = public_data.get('property_price_trends', {})
        features['avg_price_trend'] = np.mean(list(price_trends.values())) if price_trends else 0.0
        features['max_price_trend'] = max(price_trends.values()) if price_trends else 0.0
        features['price_trend_variance'] = np.var(list(price_trends.values())) if price_trends else 0.0
        
        features['area_development_score'] = float(public_data.get('area_development_score', 0))
        
        market_activity_mapping = {'low': 1, 'medium': 2, 'high': 3}
        features['market_activity_level'] = float(market_activity_mapping.get(
            public_data.get('market_activity_level', 'medium'), 2
        ))
        
        return features
    
    def _extract_third_party_features(self, third_party_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract third-party data features"""
        features = {}
        
        features['credit_inquiry_activity'] = float(third_party_data.get('credit_inquiry_activity', False))
        features['online_ad_engagement'] = float(third_party_data.get('online_ad_engagement', 0))
        
        social_signals = third_party_data.get('social_media_signals', {})
        features['social_engagement_score'] = self._calculate_social_engagement_score(social_signals)
        
        return features
    
    def _create_interaction_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Create interaction features between different feature groups"""
        interaction_features = {}
        
        # High-value interactions
        interaction_features['income_engagement_interaction'] = (
            features.get('income_bracket_encoded', 0) * 
            features.get('engagement_score', 0)
        )
        
        interaction_features['search_specificity_frequency'] = (
            features.get('search_query_specificity', 0) * 
            features.get('search_frequency', 0)
        )
        
        interaction_features['contact_intent_score'] = (
            features.get('contact_attempts', 0) * 
            features.get('intent_velocity', 0)
        )
        
        interaction_features['financial_readiness_score'] = (
            features.get('credit_inquiry_activity', 0) * 
            features.get('buying_power_score', 0)
        )
        
        return interaction_features
    
    def _create_temporal_features(self, lead_data: Dict[str, Any]) -> Dict[str, float]:
        """Create temporal features"""
        features = {}
        
        timestamp = lead_data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.utcnow()
        
        # Time-based features
        features['hour_of_day'] = float(timestamp.hour)
        features['day_of_week'] = float(timestamp.weekday())
        features['is_weekend'] = float(timestamp.weekday() >= 5)
        features['is_business_hours'] = float(9 <= timestamp.hour <= 17)
        
        return features
    
    def _calculate_engagement_score(self, behavioral_data: Dict[str, Any]) -> float:
        """Calculate overall engagement score"""
        weights = {
            'property_views_count': 0.3,
            'time_spent_on_platform': 0.25,
            'page_depth': 0.2,
            'return_visits': 0.15,
            'contact_attempts': 0.1
        }
        
        score = 0.0
        for feature, weight in weights.items():
            value = behavioral_data.get(feature, 0)
            # Normalize to 0-1 scale using sigmoid-like function
            normalized_value = min(1.0, value / (value + 10))
            score += weight * normalized_value
        
        return score
    
    def _calculate_intent_velocity(self, behavioral_data: Dict[str, Any]) -> float:
        """Calculate intent velocity (rate of engagement increase)"""
        interaction_freq = behavioral_data.get('property_interaction_frequency', 0)
        search_freq = behavioral_data.get('search_frequency', 0)
        contact_attempts = behavioral_data.get('contact_attempts', 0)
        
        # Simple velocity calculation based on recent activity
        velocity = (interaction_freq * 0.4 + search_freq * 0.4 + contact_attempts * 0.2) / 30
        return min(1.0, velocity)
    
    def _calculate_platform_stickiness(self, behavioral_data: Dict[str, Any]) -> float:
        """Calculate platform stickiness (retention indicator)"""
        return_visits = behavioral_data.get('return_visits', 0)
        time_spent = behavioral_data.get('time_spent_on_platform', 0)
        
        if return_visits == 0:
            return 0.0
        
        avg_session_time = time_spent / max(1, return_visits)
        stickiness = min(1.0, (avg_session_time * return_visits) / 100)
        return stickiness
    
    def _calculate_search_intensity(self, behavioral_data: Dict[str, Any]) -> float:
        """Calculate search intensity and specificity combined"""
        search_freq = behavioral_data.get('search_frequency', 0)
        specificity = behavioral_data.get('search_query_specificity', 0)
        
        intensity = (search_freq / 30) * specificity  # Normalize by 30 days
        return min(1.0, intensity)
    
    def _calculate_interaction_depth(self, behavioral_data: Dict[str, Any]) -> float:
        """Calculate depth of interaction with platform"""
        page_depth = behavioral_data.get('page_depth', 0)
        property_views = behavioral_data.get('property_views_count', 0)
        
        if property_views == 0:
            return 0.0
        
        depth_score = (page_depth / property_views) / 10  # Normalize
        return min(1.0, depth_score)
    
    def _calculate_buying_power_score(self, demographic_data: Dict[str, Any]) -> float:
        """Calculate buying power score from demographic data"""
        income_scores = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'premium': 1.0}
        age_scores = {'18-25': 0.3, '26-35': 0.8, '36-45': 1.0, '46-60': 0.9, '60+': 0.7}
        
        income_score = income_scores.get(demographic_data.get('income_bracket', '').lower(), 0)
        age_score = age_scores.get(demographic_data.get('age_range', ''), 0.5)
        
        return (income_score * 0.7 + age_score * 0.3)
    
    def _calculate_life_stage_score(self, demographic_data: Dict[str, Any]) -> float:
        """Calculate life stage score indicating property purchase likelihood"""
        family_scores = {
            'single': 0.4, 'couple': 0.7, 
            'family_with_children': 0.9, 'joint_family': 0.8
        }
        
        age_multipliers = {'18-25': 0.6, '26-35': 1.0, '36-45': 0.9, '46-60': 0.7, '60+': 0.5}
        
        family_score = family_scores.get(demographic_data.get('family_composition', '').lower(), 0.5)
        age_multiplier = age_multipliers.get(demographic_data.get('age_range', ''), 0.8)
        
        return family_score * age_multiplier
    
    def _calculate_social_engagement_score(self, social_signals: Dict[str, Any]) -> float:
        """Calculate social media engagement score"""
        # Simplified calculation - in reality would use more sophisticated NLP
        engagement_indicators = ['property_posts', 'real_estate_follows', 'location_checkins']
        
        score = 0.0
        for indicator in engagement_indicators:
            if indicator in social_signals:
                score += min(1.0, social_signals[indicator] / 10)
        
        return score / len(engagement_indicators)

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names for model training"""
        # This would return all possible feature names
        # For brevity, returning a subset
        return [
            'property_interaction_frequency', 'search_query_specificity',
            'time_spent_minutes', 'property_views_count', 'search_frequency',
            'page_depth', 'return_visits', 'contact_attempts',
            'engagement_score', 'intent_velocity', 'platform_stickiness',
            'search_intensity', 'interaction_depth',
            'income_bracket_encoded', 'family_composition_encoded', 'age_range_encoded',
            'buying_power_score', 'life_stage_score', 'area_preference_diversity',
            'avg_price_trend', 'max_price_trend', 'price_trend_variance',
            'area_development_score', 'market_activity_level',
            'credit_inquiry_activity', 'online_ad_engagement', 'social_engagement_score',
            'income_engagement_interaction', 'search_specificity_frequency',
            'contact_intent_score', 'financial_readiness_score',
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hours'
        ]
