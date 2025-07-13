import os
from functools import lru_cache
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    
    # Database
    database_url: str = "postgresql://username:password@localhost:5432/lead_scoring"
    redis_url: str = "redis://localhost:6379"
    
    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_behavioral: str = "behavioral_data"
    kafka_topic_demographic: str = "demographic_data"
    
    # Groq API
    groq_api_key: str = ""
    groq_model: str = "llama3-8b-8192"
    
    # CRM Integration
    crm_api_url: str = ""
    crm_api_key: str = ""
    whatsapp_api_url: str = ""
    whatsapp_api_key: str = ""
    
    # Model Configuration
    model_retrain_interval_hours: int = 24
    feature_drift_threshold: float = 0.05
    max_latency_ms: int = 300
    
    # Security
    secret_key: str = "dev-secret-key"
    encryption_key: str = "dev-encryption-key"
    
    # Monitoring
    prometheus_port: int = 9090
    log_level: str = "INFO"
    
    # Compliance
    data_retention_days: int = 365
    consent_required: bool = True
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
