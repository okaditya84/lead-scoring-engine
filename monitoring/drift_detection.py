import asyncio
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from scipy import stats
import redis.asyncio as redis
from sqlalchemy import create_engine, text
import schedule
import time
from threading import Thread

from config import get_settings
from models.lead_scoring_model import LeadScoringModel
from services.integration_service import IntegrationService

logger = logging.getLogger(__name__)

class DriftDetector:
    """Detects feature drift and model performance degradation"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self.redis_available = False
        self.reference_data = None
        self.drift_thresholds = {
            'ks_test': 0.05,  # Kolmogorov-Smirnov test p-value threshold
            'psi': 0.1,       # Population Stability Index threshold
            'accuracy_drop': 0.05  # Acceptable accuracy drop
        }
    
    async def initialize(self):
        """Initialize Redis connection and load reference data"""
        try:
            self.redis_client = redis.from_url(self.settings.redis_url)
            # Test connection
            await self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis connection established")
            await self.load_reference_data()
        except Exception as e:
            logger.warning(f"Redis not available: {str(e)}. Running in fallback mode.")
            self.redis_available = False
            self.redis_client = None
    
    async def load_reference_data(self):
        """Load reference data for drift detection"""
        if not self.redis_available:
            logger.warning("Redis not available. Using fallback reference data.")
            # Create dummy reference data for fallback
            self.reference_data = {
                'feature_stats': {
                    'annual_revenue': {'mean': 500000, 'std': 200000},
                    'company_size': {'mean': 100, 'std': 50},
                    'engagement_score': {'mean': 0.5, 'std': 0.2}
                },
                'timestamp': datetime.now().isoformat()
            }
            return
            
        try:
            reference_data_str = await self.redis_client.get("reference_data")
            if reference_data_str:
                self.reference_data = json.loads(reference_data_str)
                logger.info("Reference data loaded from Redis")
            else:
                logger.warning("No reference data found in Redis")
        except Exception as e:
            logger.error(f"Failed to load reference data: {str(e)}")
    
    async def detect_feature_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect drift in feature distributions"""
        if self.reference_data is None:
            logger.warning("No reference data available for drift detection")
            return {'status': 'no_reference_data'}
        
        drift_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'drift_detected': False,
            'feature_drifts': {},
            'overall_drift_score': 0.0
        }
        
        try:
            for feature in current_data.columns:
                if feature in self.reference_data:
                    drift_score = await self._calculate_drift_score(
                        current_data[feature].values,
                        self.reference_data[feature]
                    )
                    
                    drift_results['feature_drifts'][feature] = drift_score
                    
                    if drift_score['ks_pvalue'] < self.drift_thresholds['ks_test']:
                        drift_results['drift_detected'] = True
            
            # Calculate overall drift score
            if drift_results['feature_drifts']:
                psi_scores = [d['psi'] for d in drift_results['feature_drifts'].values()]
                drift_results['overall_drift_score'] = np.mean(psi_scores)
            
            # Store results in Redis if available
            if self.redis_available:
                try:
                    await self.redis_client.setex(
                        "latest_drift_detection",
                        3600,  # 1 hour TTL
                        json.dumps(drift_results)
                    )
                except Exception as e:
                    logger.warning(f"Failed to store drift results in Redis: {str(e)}")
            
            return drift_results
            
        except Exception as e:
            logger.error(f"Drift detection failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def _calculate_drift_score(self, current_values: np.ndarray, reference_values: List[float]) -> Dict[str, float]:
        """Calculate drift score between current and reference distributions"""
        reference_array = np.array(reference_values)
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(current_values, reference_array)
        
        # Population Stability Index (PSI)
        psi = self._calculate_psi(current_values, reference_array)
        
        return {
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'psi': psi,
            'mean_shift': float(np.mean(current_values) - np.mean(reference_array)),
            'std_shift': float(np.std(current_values) - np.std(reference_array))
        }
    
    def _calculate_psi(self, current: np.ndarray, reference: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference, bins=bins)
            
            # Calculate distributions
            current_dist, _ = np.histogram(current, bins=bin_edges)
            reference_dist, _ = np.histogram(reference, bins=bin_edges)
            
            # Normalize to get proportions
            current_prop = current_dist / len(current)
            reference_prop = reference_dist / len(reference)
            
            # Add small epsilon to avoid division by zero
            epsilon = 1e-6
            current_prop = np.maximum(current_prop, epsilon)
            reference_prop = np.maximum(reference_prop, epsilon)
            
            # Calculate PSI
            psi = np.sum((current_prop - reference_prop) * np.log(current_prop / reference_prop))
            
            return float(psi)
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {str(e)}")
            return 0.0
    
    async def update_reference_data(self, new_data: pd.DataFrame):
        """Update reference data with new training data"""
        try:
            reference_dict = {}
            for column in new_data.columns:
                if new_data[column].dtype in [np.float64, np.int64]:
                    reference_dict[column] = new_data[column].tolist()
            
            if self.redis_available:
                try:
                    await self.redis_client.set("reference_data", json.dumps(reference_dict))
                except Exception as e:
                    logger.warning(f"Failed to store reference data in Redis: {str(e)}")
                    
            self.reference_data = reference_dict
            logger.info("Reference data updated")
            
        except Exception as e:
            logger.error(f"Failed to update reference data: {str(e)}")

class PerformanceMonitor:
    """Monitors model performance and triggers retraining"""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = None
        self.redis_available = False
        self.integration_service = IntegrationService()
        self.performance_history = []
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.settings.redis_url)
            await self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis connection established for PerformanceMonitor")
        except Exception as e:
            logger.warning(f"Redis not available for PerformanceMonitor: {str(e)}. Running in fallback mode.")
            self.redis_available = False
            self.redis_client = None
    
    async def track_prediction(self, lead_id: str, prediction: Dict[str, Any]):
        """Track a prediction for later performance evaluation"""
        if not self.redis_available:
            logger.warning("Redis not available. Prediction tracking disabled.")
            return
            
        tracking_data = {
            'lead_id': lead_id,
            'predicted_score': prediction['final_score'],
            'prediction_timestamp': prediction['timestamp'],
            'features': prediction.get('feature_scores', {}),
            'model_version': prediction.get('model_version', 'unknown')
        }
        
        try:
            # Store in Redis with 30-day TTL
            await self.redis_client.setex(
                f"prediction:{lead_id}",
                30 * 24 * 3600,  # 30 days
                json.dumps(tracking_data)
            )
        except Exception as e:
            logger.warning(f"Failed to track prediction in Redis: {str(e)}")
    
    async def record_conversion(self, lead_id: str, converted: bool, conversion_value: Optional[float] = None):
        """Record actual conversion outcome"""
        if not self.redis_available:
            logger.warning("Redis not available. Conversion tracking disabled.")
            return
            
        try:
            # Get original prediction
            prediction_data = await self.redis_client.get(f"prediction:{lead_id}")
            if not prediction_data:
                logger.warning(f"No prediction found for lead {lead_id}")
                return
            
            prediction = json.loads(prediction_data)
            
            # Calculate performance metrics
            performance_data = {
                'lead_id': lead_id,
                'predicted_score': prediction['predicted_score'],
                'actual_conversion': converted,
                'conversion_value': conversion_value,
                'prediction_timestamp': prediction['prediction_timestamp'],
                'conversion_timestamp': datetime.utcnow().isoformat(),
                'prediction_accuracy': self._calculate_accuracy(
                    prediction['predicted_score'], converted
                )
            }
            
            # Store performance data
            try:
                await self.redis_client.lpush(
                    "performance_history",
                    json.dumps(performance_data)
                )
                
                # Keep only last 10000 records
                await self.redis_client.ltrim("performance_history", 0, 9999)
            except Exception as e:
                logger.warning(f"Failed to store performance data in Redis: {str(e)}")
            
            logger.info(f"Recorded conversion for lead {lead_id}: {converted}")
            
        except Exception as e:
            logger.error(f"Failed to record conversion: {str(e)}")
    
    def _calculate_accuracy(self, predicted_score: float, actual_conversion: bool) -> float:
        """Calculate prediction accuracy"""
        # Convert score to binary prediction (threshold at 50)
        predicted_conversion = predicted_score > 50
        return 1.0 if predicted_conversion == actual_conversion else 0.0
    
    async def calculate_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """Calculate performance metrics for recent predictions"""
        if not self.redis_available:
            logger.warning("Redis not available. Cannot calculate performance metrics.")
            return {'status': 'redis_unavailable'}
            
        try:
            # Get recent performance data
            performance_data = await self.redis_client.lrange("performance_history", 0, -1)
            
            if not performance_data:
                return {'status': 'no_data'}
            
            recent_records = []
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            for record in performance_data:
                data = json.loads(record)
                conversion_time = datetime.fromisoformat(data['conversion_timestamp'])
                if conversion_time >= cutoff_date:
                    recent_records.append(data)
            
            if not recent_records:
                return {'status': 'no_recent_data'}
            
            # Calculate metrics
            accuracies = [r['prediction_accuracy'] for r in recent_records]
            predicted_scores = [r['predicted_score'] for r in recent_records]
            actual_conversions = [r['actual_conversion'] for r in recent_records]
            
            metrics = {
                'period_days': days,
                'total_predictions': len(recent_records),
                'accuracy': np.mean(accuracies),
                'conversion_rate': np.mean(actual_conversions),
                'average_predicted_score': np.mean(predicted_scores),
                'score_correlation': np.corrcoef(predicted_scores, actual_conversions)[0, 1] if len(recent_records) > 1 else 0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Store latest metrics if Redis is available
            try:
                await self.redis_client.setex(
                    "latest_performance_metrics",
                    3600,  # 1 hour TTL
                    json.dumps(metrics)
                )
            except Exception as e:
                logger.warning(f"Failed to store performance metrics in Redis: {str(e)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance calculation failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def check_performance_degradation(self) -> bool:
        """Check if model performance has degraded significantly"""
        try:
            current_metrics = await self.calculate_recent_performance(days=7)
            baseline_metrics = await self.calculate_recent_performance(days=30)
            
            if (current_metrics.get('status') != 'no_data' and 
                baseline_metrics.get('status') != 'no_data'):
                
                accuracy_drop = (baseline_metrics['accuracy'] - 
                               current_metrics['accuracy'])
                
                if accuracy_drop > self.settings.feature_drift_threshold:
                    logger.warning(f"Performance degradation detected: {accuracy_drop:.3f}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Performance degradation check failed: {str(e)}")
            return False

class ContinuousLearning:
    """Manages continuous learning and model retraining"""
    
    def __init__(self):
        self.settings = get_settings()
        self.model = LeadScoringModel()
        self.drift_detector = DriftDetector()
        self.performance_monitor = PerformanceMonitor()
        self.redis_client = None
        self.redis_available = False
        self.db_engine = None
        self._scheduler_running = False
    
    async def initialize(self):
        """Initialize all components"""
        try:
            self.redis_client = redis.from_url(self.settings.redis_url)
            await self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis connection established for ContinuousLearning")
        except Exception as e:
            logger.warning(f"Redis not available for ContinuousLearning: {str(e)}. Running in fallback mode.")
            self.redis_available = False
            self.redis_client = None
            
        # Database connection (optional for sample data)
        try:
            self.db_engine = create_engine(self.settings.database_url)
        except Exception as e:
            logger.warning(f"Database not available: {str(e)}. Using sample data fallback.")
            self.db_engine = None
        
        await self.drift_detector.initialize()
        await self.performance_monitor.initialize()
        
        # Start scheduler in background
        self._start_scheduler()
    
    def _start_scheduler(self):
        """Start background scheduler for automated tasks"""
        if self._scheduler_running:
            return
        
        # Schedule daily model health checks
        schedule.every().day.at("02:00").do(self._run_daily_checks)
        
        # Schedule weekly retraining
        schedule.every().week.do(self._run_weekly_retrain)
        
        def run_scheduler():
            while self._scheduler_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self._scheduler_running = True
        scheduler_thread = Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        
        logger.info("Continuous learning scheduler started")
    
    async def _run_daily_checks(self):
        """Run daily health checks"""
        try:
            logger.info("Running daily health checks...")
            
            # Check for feature drift
            recent_data = await self._get_recent_feature_data()
            if recent_data is not None:
                drift_results = await self.drift_detector.detect_feature_drift(recent_data)
                
                if drift_results.get('drift_detected'):
                    logger.warning("Feature drift detected - scheduling retrain")
                    await self._trigger_retrain()
            
            # Check performance degradation
            performance_degraded = await self.performance_monitor.check_performance_degradation()
            if performance_degraded:
                logger.warning("Performance degradation detected - scheduling retrain")
                await self._trigger_retrain()
            
        except Exception as e:
            logger.error(f"Daily checks failed: {str(e)}")
    
    async def _run_weekly_retrain(self):
        """Run weekly model retraining"""
        try:
            logger.info("Running weekly model retraining...")
            await self._trigger_retrain()
        except Exception as e:
            logger.error(f"Weekly retrain failed: {str(e)}")
    
    async def _get_recent_feature_data(self, days: int = 7) -> Optional[pd.DataFrame]:
        """Get recent feature data for drift detection"""
        try:
            # This would typically query your data warehouse
            # For now, we'll simulate by getting data from Redis
            
            # In a real implementation, you'd query your database:
            # query = f"""
            # SELECT * FROM lead_features 
            # WHERE created_at >= NOW() - INTERVAL '{days} days'
            # """
            # return pd.read_sql(query, self.db_engine)
            
            # Placeholder implementation
            logger.info(f"Getting recent feature data for {days} days")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get recent feature data: {str(e)}")
            return None
    
    async def _trigger_retrain(self):
        """Trigger model retraining"""
        try:
            logger.info("Triggering model retraining...")
            
            # Get updated training data
            training_data = await self._get_training_data()
            if training_data is None or len(training_data) < 100:
                logger.warning("Insufficient training data for retraining")
                return
            
            # Retrain model
            training_results = self.model.train(training_data)
            
            # Update reference data for drift detection
            feature_data = self.model._prepare_features(training_data)
            await self.drift_detector.update_reference_data(feature_data)
            
            # Save updated model
            model_path = f"models/lead_scoring_model_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib"
            self.model.save_model(model_path)
            
            # Store retraining info
            retrain_info = {
                'timestamp': datetime.utcnow().isoformat(),
                'training_samples': len(training_data),
                'model_path': model_path,
                'performance': training_results
            }
            
            # Store retraining info if Redis is available
            if self.redis_available:
                try:
                    await self.redis_client.setex(
                        "latest_retrain_info",
                        7 * 24 * 3600,  # 7 days TTL
                        json.dumps(retrain_info)
                    )
                except Exception as e:
                    logger.warning(f"Failed to store retrain info in Redis: {str(e)}")
            
            logger.info(f"Model retraining completed. New model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Model retraining failed: {str(e)}")
    
    async def _get_training_data(self) -> Optional[pd.DataFrame]:
        """Get training data with conversion outcomes"""
        try:
            # In a real implementation, this would query your database
            # for leads with conversion outcomes
            
            # Placeholder for database query
            # query = """
            # SELECT l.*, c.converted, c.conversion_value
            # FROM leads l
            # LEFT JOIN conversions c ON l.lead_id = c.lead_id
            # WHERE l.created_at >= NOW() - INTERVAL '90 days'
            # AND c.conversion_timestamp IS NOT NULL
            # """
            # return pd.read_sql(query, self.db_engine)
            
            logger.info("Getting training data...")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get training data: {str(e)}")
            return None
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        try:
            drift_info = {}
            performance_info = {}
            retrain_info = {}
            
            if self.redis_available:
                # Get latest drift detection results
                drift_data = await self.redis_client.get("latest_drift_detection")
                drift_info = json.loads(drift_data) if drift_data else {}
                
                # Get latest performance metrics
                performance_data = await self.redis_client.get("latest_performance_metrics")
                performance_info = json.loads(performance_data) if performance_data else {}
                
                # Get latest retrain info
                retrain_data = await self.redis_client.get("latest_retrain_info")
                retrain_info = json.loads(retrain_data) if retrain_data else {}
            else:
                logger.warning("Redis not available. Health check using fallback data.")
            
            # Determine overall health status
            health_status = "healthy"
            if drift_info.get('drift_detected'):
                health_status = "degraded"
            if performance_info.get('accuracy', 1.0) < 0.7:
                health_status = "unhealthy"
            
            return {
                'status': health_status,
                'timestamp': datetime.utcnow().isoformat(),
                'drift_detection': drift_info,
                'performance_metrics': performance_info,
                'last_retrain': retrain_info,
                'model_trained': self.model.is_trained,
                'redis_available': self.redis_available
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    async def close(self):
        """Clean up resources"""
        self._scheduler_running = False
        if self.redis_client:
            await self.redis_client.close()
        if self.db_engine:
            self.db_engine.dispose()
