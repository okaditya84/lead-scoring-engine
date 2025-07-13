from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class PropertyType(str, Enum):
    APARTMENT = "apartment"
    VILLA = "villa"
    PLOT = "plot"
    COMMERCIAL = "commercial"

class LeadSource(str, Enum):
    WEBSITE = "website"
    WHATSAPP = "whatsapp"
    SOCIAL_MEDIA = "social_media"
    REFERRAL = "referral"
    ADVERTISEMENT = "advertisement"

class BehavioralData(BaseModel):
    """Behavioral features for lead scoring"""
    property_interaction_frequency: int = Field(..., description="Number of property interactions in last 30 days")
    search_query_specificity: float = Field(..., description="Score 0-1 indicating search specificity")
    time_spent_on_platform: int = Field(..., description="Total time spent in minutes")
    property_views_count: int = Field(..., description="Number of properties viewed")
    search_frequency: int = Field(..., description="Number of searches performed")
    page_depth: int = Field(..., description="Average pages viewed per session")
    return_visits: int = Field(..., description="Number of return visits")
    contact_attempts: int = Field(..., description="Number of contact form submissions/calls")

class DemographicData(BaseModel):
    """Demographic features for lead scoring"""
    income_bracket: Optional[str] = Field(None, description="Income range: low/medium/high/premium")
    family_composition: Optional[str] = Field(None, description="single/couple/family_with_children/joint_family")
    age_range: Optional[str] = Field(None, description="18-25/26-35/36-45/46-60/60+")
    occupation: Optional[str] = Field(None, description="Professional category")
    location: Optional[str] = Field(None, description="Current location/city")
    preferred_areas: List[str] = Field(default=[], description="Areas of interest")

class PublicData(BaseModel):
    """Public data features"""
    property_price_trends: Dict[str, float] = Field(default={}, description="Price trends in searched areas")
    area_development_score: float = Field(0.0, description="Development score of searched areas")
    market_activity_level: str = Field("medium", description="low/medium/high market activity")

class ThirdPartyData(BaseModel):
    """Third-party data features"""
    credit_inquiry_activity: bool = Field(False, description="Recent credit inquiries")
    online_ad_engagement: float = Field(0.0, description="Engagement score with real estate ads")
    social_media_signals: Dict[str, Any] = Field(default={}, description="Social media activity signals")

class LeadData(BaseModel):
    """Complete lead data structure"""
    lead_id: str = Field(..., description="Unique lead identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: LeadSource = Field(..., description="Lead source")
    
    # Core data groups
    behavioral: BehavioralData
    demographic: DemographicData
    public_data: PublicData
    third_party: ThirdPartyData
    
    # Contact information (encrypted)
    contact_info: Dict[str, str] = Field(default={}, description="Encrypted contact information")
    
    # Consent and compliance
    consent_given: bool = Field(False, description="Data processing consent")
    consent_timestamp: Optional[datetime] = None

class LeadScore(BaseModel):
    """Lead scoring result"""
    lead_id: str
    base_score: float = Field(..., description="GBT model score (0-1)")
    llm_adjusted_score: float = Field(..., description="LLM re-ranked score (0-1)")
    final_score: float = Field(..., description="Final composite score (0-100)")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    reasoning: str = Field(..., description="LLM reasoning for score adjustment")
    priority: str = Field(..., description="high/medium/low priority")
    recommendations: List[str] = Field(default=[], description="Action recommendations")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BatchScoreRequest(BaseModel):
    """Batch scoring request"""
    leads: List[LeadData]
    priority_threshold: float = Field(0.7, description="Threshold for high priority classification")

class BatchScoreResponse(BaseModel):
    """Batch scoring response"""
    scores: List[LeadScore]
    processing_time_ms: float
    total_leads: int
    high_priority_count: int

class ModelHealth(BaseModel):
    """Model health status"""
    status: str = Field(..., description="healthy/degraded/unhealthy")
    last_retrain: datetime
    drift_score: float
    latency_p95: float
    accuracy_score: float
    features_status: Dict[str, str]

class ConversionFeedback(BaseModel):
    """Conversion feedback for model improvement"""
    lead_id: str
    predicted_score: float
    actual_conversion: bool
    conversion_value: Optional[float] = None
    conversion_timestamp: datetime
    feedback_notes: Optional[str] = None
