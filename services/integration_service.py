import asyncio
import json
import httpx
import logging
from typing import Dict, Any, List
from datetime import datetime

from config import get_settings

logger = logging.getLogger(__name__)

class IntegrationService:
    """Service for CRM and WhatsApp integrations"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def notify_crm(self, lead_score: Dict[str, Any]) -> bool:
        """Send lead score to CRM system"""
        if not self.settings.crm_api_url or not self.settings.crm_api_key:
            logger.warning("CRM configuration not found, skipping notification")
            return False
        
        try:
            payload = {
                'lead_id': lead_score['lead_id'],
                'score': lead_score['final_score'],
                'priority': lead_score['priority'],
                'recommendations': lead_score['recommendations'],
                'reasoning': lead_score['reasoning'],
                'timestamp': lead_score['timestamp'],
                'confidence': lead_score['confidence']
            }
            
            headers = {
                'Authorization': f'Bearer {self.settings.crm_api_key}',
                'Content-Type': 'application/json'
            }
            
            response = await self.client.post(
                f"{self.settings.crm_api_url}/leads/update-score",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully updated CRM for lead {lead_score['lead_id']}")
                return True
            else:
                logger.error(f"CRM update failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"CRM integration error: {str(e)}")
            return False
    
    async def notify_whatsapp(self, lead_score: Dict[str, Any], agent_phone: str) -> bool:
        """Send WhatsApp notification to assigned agent"""
        if not self.settings.whatsapp_api_url or not self.settings.whatsapp_api_key:
            logger.warning("WhatsApp configuration not found, skipping notification")
            return False
        
        try:
            # Generate WhatsApp message
            message = self._generate_whatsapp_message(lead_score)
            
            payload = {
                'phone': agent_phone,
                'message': message,
                'lead_id': lead_score['lead_id']
            }
            
            headers = {
                'Authorization': f'Bearer {self.settings.whatsapp_api_key}',
                'Content-Type': 'application/json'
            }
            
            response = await self.client.post(
                f"{self.settings.whatsapp_api_url}/send-message",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                logger.info(f"WhatsApp notification sent for lead {lead_score['lead_id']}")
                return True
            else:
                logger.error(f"WhatsApp notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"WhatsApp integration error: {str(e)}")
            return False
    
    def _generate_whatsapp_message(self, lead_score: Dict[str, Any]) -> str:
        """Generate WhatsApp message for agent notification"""
        priority_emoji = {
            'high': '🔴',
            'medium': '🟡', 
            'low': '🟢'
        }
        
        emoji = priority_emoji.get(lead_score['priority'], '⚪')
        
        message = f"""{emoji} *Lead Alert - {lead_score['priority'].upper()} Priority*

🆔 Lead ID: {lead_score['lead_id']}
⭐ Score: {lead_score['final_score']:.0f}/100
🎯 Confidence: {lead_score['confidence']:.0%}

📝 *Reasoning:*
{lead_score['reasoning']}

🎯 *Recommendations:*
"""
        
        for i, rec in enumerate(lead_score['recommendations'][:3], 1):
            message += f"{i}. {rec}\n"
        
        message += f"\n⏰ Generated: {datetime.fromisoformat(lead_score['timestamp']).strftime('%H:%M %d/%m')}"
        
        return message
    
    async def get_lead_conversion_data(self, lead_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch conversion data from CRM for model feedback"""
        if not self.settings.crm_api_url or not self.settings.crm_api_key:
            logger.warning("CRM configuration not found")
            return {}
        
        try:
            headers = {
                'Authorization': f'Bearer {self.settings.crm_api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {'lead_ids': lead_ids}
            
            response = await self.client.post(
                f"{self.settings.crm_api_url}/leads/conversion-data",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch conversion data: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching conversion data: {str(e)}")
            return {}
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

class NotificationService:
    """Service for managing all notifications"""
    
    def __init__(self):
        self.integration_service = IntegrationService()
    
    async def notify_high_priority_lead(self, lead_score: Dict[str, Any], agent_assignments: Dict[str, str]) -> Dict[str, bool]:
        """Notify about high priority leads via multiple channels"""
        results = {}
        
        # CRM notification
        results['crm'] = await self.integration_service.notify_crm(lead_score)
        
        # WhatsApp notification for high-priority leads
        if lead_score['priority'] == 'high':
            agent_phone = agent_assignments.get(lead_score['lead_id'])
            if agent_phone:
                results['whatsapp'] = await self.integration_service.notify_whatsapp(lead_score, agent_phone)
            else:
                results['whatsapp'] = False
                logger.warning(f"No agent assigned for high-priority lead {lead_score['lead_id']}")
        
        return results
    
    async def bulk_notify(self, lead_scores: List[Dict[str, Any]], agent_assignments: Dict[str, str]) -> List[Dict[str, Any]]:
        """Send bulk notifications for multiple leads"""
        notification_results = []
        
        for lead_score in lead_scores:
            result = await self.notify_high_priority_lead(lead_score, agent_assignments)
            notification_results.append({
                'lead_id': lead_score['lead_id'],
                'notifications_sent': result
            })
        
        return notification_results
    
    async def close(self):
        """Clean up resources"""
        await self.integration_service.close()
