from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Try to import pandas, fallback if not available
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Try to import structlog, fallback to standard logging
try:
    import structlog
except ImportError:
    structlog = logging

from config import get_settings
from models.schemas import (
    LeadData, LeadScore, BatchScoreRequest, BatchScoreResponse,
    ModelHealth, ConversionFeedback
)
from models.lead_scoring_model import LeadScoringModel
from services.integration_service import NotificationService
from monitoring.drift_detection import ContinuousLearning

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
prediction_counter = Counter('lead_scoring_predictions_total', 'Total predictions made')
prediction_latency = Histogram('lead_scoring_prediction_duration_seconds', 'Prediction latency')
model_accuracy = Gauge('lead_scoring_model_accuracy', 'Current model accuracy')
drift_score = Gauge('lead_scoring_drift_score', 'Current drift score')
high_priority_leads = Counter('lead_scoring_high_priority_total', 'High priority leads identified')

# Global instances
settings = get_settings()

app = FastAPI(
    title="Lead Scoring Engine",
    description="Real Estate Lead Scoring API with GBT + LLM Re-ranker",
    version="1.0.0",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None
)

lead_scoring_model = LeadScoringModel()
notification_service = NotificationService()
continuous_learning = ContinuousLearning()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [settings.frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        logger.info("Starting Lead Scoring Engine...")
        
        # Start Prometheus metrics server
        start_http_server(settings.prometheus_port)
        logger.info(f"Prometheus metrics server started on port {settings.prometheus_port}")
        
        # Initialize continuous learning system
        await continuous_learning.initialize()
        
        # Load pre-trained model if available
        try:
            lead_scoring_model.load_model("models/lead_scoring_model.joblib")
            logger.info("Pre-trained model loaded successfully")
        except Exception as e:
            logger.warning(f"No pre-trained model found: {str(e)}")
            # Initialize with sample data for demo
            await _initialize_demo_model()
        
        logger.info("Lead Scoring Engine started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    try:
        await notification_service.close()
        await continuous_learning.close()
        logger.info("Lead Scoring Engine shutdown completed")
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

async def _initialize_demo_model():
    """Initialize model with demo data for testing"""
    try:
        logger.info("Initializing demo model...")
        
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        demo_data = []
        for i in range(n_samples):
            # Generate synthetic lead data
            lead_data = {
                'lead_id': f'demo_{i}',
                'behavioral': {
                    'property_interaction_frequency': np.random.randint(1, 50),
                    'search_query_specificity': np.random.random(),
                    'time_spent_on_platform': np.random.randint(5, 300),
                    'property_views_count': np.random.randint(1, 20),
                    'search_frequency': np.random.randint(1, 30),
                    'page_depth': np.random.randint(1, 10),
                    'return_visits': np.random.randint(0, 15),
                    'contact_attempts': np.random.randint(0, 5)
                },
                'demographic': {
                    'income_bracket': np.random.choice(['low', 'medium', 'high', 'premium']),
                    'family_composition': np.random.choice(['single', 'couple', 'family_with_children', 'joint_family']),
                    'age_range': np.random.choice(['18-25', '26-35', '36-45', '46-60', '60+']),
                    'preferred_areas': [f'area_{j}' for j in range(np.random.randint(1, 4))]
                },
                'public_data': {
                    'property_price_trends': {f'area_{j}': np.random.uniform(-0.1, 0.2) for j in range(3)},
                    'area_development_score': np.random.random(),
                    'market_activity_level': np.random.choice(['low', 'medium', 'high'])
                },
                'third_party': {
                    'credit_inquiry_activity': np.random.choice([True, False]),
                    'online_ad_engagement': np.random.random(),
                    'social_media_signals': {}
                }
            }
            
            # Generate synthetic conversion target
            # Higher scores for more engaged leads
            engagement_score = (
                lead_data['behavioral']['contact_attempts'] * 0.3 +
                lead_data['behavioral']['property_interaction_frequency'] * 0.01 +
                lead_data['behavioral']['search_query_specificity'] * 0.4
            )
            
            conversion_prob = min(0.9, engagement_score / 3)
            converted = np.random.random() < conversion_prob
            
            lead_data['converted'] = converted
            demo_data.append(lead_data)
        
        # Train model
        if PANDAS_AVAILABLE:
            df = pd.DataFrame(demo_data)
            training_results = lead_scoring_model.train(df)
            
            # Save model
            lead_scoring_model.save_model("models/lead_scoring_model.joblib")
            
            logger.info(f"Demo model trained successfully. Accuracy: {training_results['test_accuracy']:.4f}")
        else:
            logger.warning("Pandas not available, skipping model training. Using fallback model.")
        
    except Exception as e:
        logger.error(f"Demo model initialization failed: {str(e)}")

@app.post("/score-lead", response_model=LeadScore)
async def score_lead(lead_data: LeadData, background_tasks: BackgroundTasks):
    """Score a single lead"""
    start_time = time.time()
    
    try:
        prediction_counter.inc()
        
        # Validate consent if required
        if settings.consent_required and not lead_data.consent_given:
            raise HTTPException(status_code=400, detail="Data processing consent required")
        
        # Convert to dict and score
        lead_dict = lead_data.dict()
        score_result = await lead_scoring_model.predict_single(lead_dict)
        
        # Track prediction for monitoring
        background_tasks.add_task(
            continuous_learning.performance_monitor.track_prediction,
            lead_data.lead_id,
            score_result
        )
        
        # Send notifications for high-priority leads
        if score_result['priority'] == 'high':
            high_priority_leads.inc()
            background_tasks.add_task(
                _send_high_priority_notification,
                score_result
            )
        
        # Record latency
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        
        # Validate latency requirement
        if latency > settings.max_latency_ms / 1000:
            logger.warning(f"Prediction latency exceeded threshold: {latency*1000:.1f}ms")
        
        return LeadScore(**score_result)
        
    except Exception as e:
        logger.error(f"Lead scoring failed: {str(e)}", lead_id=lead_data.lead_id)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")

@app.post("/score-batch", response_model=BatchScoreResponse)
async def score_batch(request: BatchScoreRequest, background_tasks: BackgroundTasks):
    """Score multiple leads in batch"""
    start_time = time.time()
    
    try:
        # Validate all leads have consent if required
        if settings.consent_required:
            invalid_leads = [lead.lead_id for lead in request.leads if not lead.consent_given]
            if invalid_leads:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Consent required for leads: {invalid_leads}"
                )
        
        # Convert to list of dicts
        leads_data = [lead.dict() for lead in request.leads]
        
        # Batch prediction
        score_results = await lead_scoring_model.predict_batch(leads_data)
        
        # Count high priority leads
        high_priority_count = sum(1 for result in score_results 
                                if result['priority'] == 'high')
        
        # Update metrics
        prediction_counter.inc(len(request.leads))
        high_priority_leads.inc(high_priority_count)
        
        # Track predictions for monitoring
        for result in score_results:
            background_tasks.add_task(
                continuous_learning.performance_monitor.track_prediction,
                result['lead_id'],
                result
            )
        
        # Send notifications for high-priority leads
        high_priority_results = [r for r in score_results if r['priority'] == 'high']
        if high_priority_results:
            background_tasks.add_task(
                _send_batch_notifications,
                high_priority_results
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchScoreResponse(
            scores=[LeadScore(**result) for result in score_results],
            processing_time_ms=processing_time,
            total_leads=len(request.leads),
            high_priority_count=high_priority_count
        )
        
    except Exception as e:
        logger.error(f"Batch scoring failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch scoring failed: {str(e)}")

@app.get("/model-health", response_model=ModelHealth)
async def get_model_health():
    """Get model health status"""
    try:
        health_data = await continuous_learning.get_system_health()
        
        # Update Prometheus metrics
        if 'performance_metrics' in health_data:
            accuracy = health_data['performance_metrics'].get('accuracy', 0)
            model_accuracy.set(accuracy)
        
        if 'drift_detection' in health_data:
            drift_score_value = health_data['drift_detection'].get('overall_drift_score', 0)
            drift_score.set(drift_score_value)
        
        # Convert to ModelHealth format
        return ModelHealth(
            status=health_data['status'],
            last_retrain=datetime.fromisoformat(
                health_data.get('last_retrain', {}).get('timestamp', datetime.utcnow().isoformat())
            ),
            drift_score=health_data.get('drift_detection', {}).get('overall_drift_score', 0.0),
            latency_p95=0.0,  # Would be calculated from metrics in production
            accuracy_score=health_data.get('performance_metrics', {}).get('accuracy', 0.0),
            features_status={}  # Would include individual feature health
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/conversion-feedback")
async def record_conversion_feedback(feedback: ConversionFeedback):
    """Record conversion feedback for model improvement"""
    try:
        await continuous_learning.performance_monitor.record_conversion(
            feedback.lead_id,
            feedback.actual_conversion,
            feedback.conversion_value
        )
        
        logger.info(f"Conversion feedback recorded for lead {feedback.lead_id}")
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        logger.error(f"Failed to record feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to record feedback: {str(e)}")

@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    """Manually trigger model retraining"""
    try:
        background_tasks.add_task(continuous_learning._trigger_retrain)
        return {"status": "success", "message": "Retraining triggered"}
        
    except Exception as e:
        logger.error(f"Failed to trigger retrain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to trigger retrain: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        health_data = await continuous_learning.get_system_health()
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "model_health": health_data,
            "system_status": "operational" if health_data['status'] != 'error' else "degraded"
        }
    except Exception as e:
        logger.error(f"Failed to get metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/feature-importance")
async def get_feature_importance():
    """Get model feature importance"""
    try:
        if not lead_scoring_model.is_trained:
            raise HTTPException(status_code=400, detail="Model is not trained")
        
        importance = lead_scoring_model.get_feature_importance()
        
        # Sort by importance
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        return {
            "feature_importance": sorted_importance,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

async def _send_high_priority_notification(score_result: Dict[str, Any]):
    """Send notification for high-priority lead"""
    try:
        # Mock agent assignment - in production, this would come from CRM
        agent_assignments = {score_result['lead_id']: "+1234567890"}
        
        await notification_service.notify_high_priority_lead(score_result, agent_assignments)
        
    except Exception as e:
        logger.error(f"Failed to send notification: {str(e)}")

async def _send_batch_notifications(high_priority_results: List[Dict[str, Any]]):
    """Send notifications for batch of high-priority leads"""
    try:
        # Mock agent assignments
        agent_assignments = {result['lead_id']: "+1234567890" for result in high_priority_results}
        
        await notification_service.bulk_notify(high_priority_results, agent_assignments)
        
    except Exception as e:
        logger.error(f"Failed to send batch notifications: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "model_trained": lead_scoring_model.is_trained
    }

@app.get("/")
async def serve_frontend():
    """Serve the frontend dashboard"""
    from fastapi.responses import FileResponse
    return FileResponse('frontend/index.html')

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )
