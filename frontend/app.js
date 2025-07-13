// Lead Scoring Engine Frontend Application
class LeadScoringApp {
    constructor() {
        this.apiBase = 'http://localhost:8000';
        this.initializeApp();
    }

    async initializeApp() {
        await this.checkSystemHealth();
        this.setupEventListeners();
        this.generateSampleData();
    }

    setupEventListeners() {
        // Lead scoring form
        document.getElementById('leadForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.scoreLead();
        });

        // Range inputs - show current value
        document.getElementById('searchSpecificity').addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = `Value: ${e.target.value}`;
        });
        
        document.getElementById('areaScore').addEventListener('input', (e) => {
            e.target.nextElementSibling.textContent = `Score: ${e.target.value}`;
        });

        // Generate random lead ID on page load
        this.generateLeadId();
    }

    generateLeadId() {
        const timestamp = new Date().getTime().toString().slice(-6);
        document.getElementById('leadId').value = `lead-${timestamp}`;
    }

    async checkSystemHealth() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const health = await response.json();
            
            const statusElement = document.getElementById('healthStatus');
            if (health.status === 'healthy') {
                statusElement.innerHTML = '<span class="health-indicator health-healthy"></span>System Healthy';
                statusElement.className = 'badge bg-success';
            } else {
                statusElement.innerHTML = '<span class="health-indicator health-degraded"></span>System Degraded';
                statusElement.className = 'badge bg-warning';
            }
        } catch (error) {
            const statusElement = document.getElementById('healthStatus');
            statusElement.innerHTML = '<span class="health-indicator health-unhealthy"></span>System Offline';
            statusElement.className = 'badge bg-danger';
            console.error('Health check failed:', error);
        }
    }

    generateSampleData() {
        // Populate form with sample data for quick testing
        const sampleData = {
            propertyInteractions: Math.floor(Math.random() * 20) + 1,
            timeOnPlatform: Math.floor(Math.random() * 120) + 10,
            propertyViews: Math.floor(Math.random() * 30) + 1,
            searchFrequency: Math.floor(Math.random() * 10) + 1,
            contactAttempts: Math.floor(Math.random() * 5),
            searchSpecificity: (Math.random()).toFixed(1)
        };

        // Don't auto-populate, let user choose
    }

    async scoreLead() {
        const loadingSpinner = document.getElementById('loadingSpinner');
        const scoreResults = document.getElementById('scoreResults');
        
        // Show loading
        loadingSpinner.style.display = 'block';
        scoreResults.style.display = 'none';

        try {
            const leadData = this.collectLeadData();
            console.log('Sending lead data:', leadData);

            const response = await fetch(`${this.apiBase}/score-lead`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(leadData)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            console.log('Received result:', result);
            
            this.displayScore(result);
        } catch (error) {
            console.error('Scoring failed:', error);
            this.showError('Failed to score lead. Please check if the API server is running.');
        } finally {
            loadingSpinner.style.display = 'none';
        }
    }

    collectLeadData() {
        return {
            lead_id: document.getElementById('leadId').value,
            source: document.getElementById('source').value,
            behavioral: {
                property_interaction_frequency: parseInt(document.getElementById('propertyInteractions').value),
                search_query_specificity: parseFloat(document.getElementById('searchSpecificity').value),
                time_spent_on_platform: parseInt(document.getElementById('timeOnPlatform').value),
                property_views_count: parseInt(document.getElementById('propertyViews').value),
                search_frequency: parseInt(document.getElementById('searchFrequency').value),
                page_depth: 3, // Default value
                return_visits: 2, // Default value
                contact_attempts: parseInt(document.getElementById('contactAttempts').value)
            },
            demographic: {
                income_bracket: document.getElementById('incomeBracket').value,
                age_range: document.getElementById('ageRange').value,
                family_composition: document.getElementById('familyComposition').value,
                occupation: document.getElementById('occupation').value,
                location: document.getElementById('location').value,
                preferred_areas: [document.getElementById('location').value.toLowerCase()]
            },
            public_data: {
                property_price_trends: {},
                area_development_score: parseFloat(document.getElementById('areaScore').value),
                market_activity_level: document.getElementById('marketActivity').value
            },
            third_party: {
                credit_inquiry_activity: document.getElementById('incomeBracket').value === 'high' || document.getElementById('incomeBracket').value === 'premium',
                online_ad_engagement: Math.random() * 0.8,
                social_media_signals: { real_estate_engagement: Math.random() * 0.6 }
            },
            consent_given: true
        };
    }

    displayScore(result) {
        const scoreResults = document.getElementById('scoreResults');
        
        // Update score display
        document.getElementById('finalScore').textContent = Math.round(result.final_score);
        document.getElementById('confidence').textContent = Math.round(result.confidence * 100);
        document.getElementById('baseScore').textContent = result.base_score.toFixed(3);
        document.getElementById('llmScore').textContent = result.llm_adjusted_score.toFixed(3);
        document.getElementById('reasoning').textContent = result.reasoning;

        // Update priority badge
        const priorityBadge = document.getElementById('priorityBadge');
        priorityBadge.textContent = result.priority.toUpperCase();
        priorityBadge.className = `priority-badge priority-${result.priority.toLowerCase()}`;

        // Update recommendations
        const recommendationsDiv = document.getElementById('recommendations');
        if (result.recommendations && result.recommendations.length > 0) {
            recommendationsDiv.innerHTML = result.recommendations
                .map(rec => `<span class="badge bg-info me-1 mb-1">${rec}</span>`)
                .join('');
        } else {
            recommendationsDiv.innerHTML = '<span class="text-muted">No specific recommendations</span>';
        }

        // Show results with animation
        scoreResults.style.display = 'block';
        scoreResults.style.opacity = '0';
        setTimeout(() => {
            scoreResults.style.opacity = '1';
            scoreResults.style.transition = 'opacity 0.5s ease-in-out';
        }, 100);
    }

    showError(message) {
        const scoreResults = document.getElementById('scoreResults');
        scoreResults.innerHTML = `
            <div class="alert alert-danger" role="alert">
                <i class="bi bi-exclamation-triangle"></i> ${message}
            </div>
        `;
        scoreResults.style.display = 'block';
    }

    // Batch processing functions
    async generateSampleBatch() {
        const batchData = [];
        const sampleNames = ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eva Brown'];
        
        for (let i = 0; i < 5; i++) {
            const lead = {
                lead_id: `batch-${Date.now()}-${i}`,
                source: ['website', 'social_media', 'referral'][Math.floor(Math.random() * 3)],
                behavioral: {
                    property_interaction_frequency: Math.floor(Math.random() * 20) + 1,
                    search_query_specificity: Math.random(),
                    time_spent_on_platform: Math.floor(Math.random() * 120) + 10,
                    property_views_count: Math.floor(Math.random() * 30) + 1,
                    search_frequency: Math.floor(Math.random() * 10) + 1,
                    page_depth: Math.floor(Math.random() * 5) + 1,
                    return_visits: Math.floor(Math.random() * 8) + 1,
                    contact_attempts: Math.floor(Math.random() * 5)
                },
                demographic: {
                    income_bracket: ['low', 'medium', 'high', 'premium'][Math.floor(Math.random() * 4)],
                    age_range: ['18-25', '26-35', '36-45', '46-55'][Math.floor(Math.random() * 4)],
                    family_composition: ['single', 'couple', 'family_with_children'][Math.floor(Math.random() * 3)],
                    occupation: 'professional',
                    location: ['Mumbai', 'Delhi', 'Bangalore', 'Pune'][Math.floor(Math.random() * 4)]
                },
                public_data: {
                    property_price_trends: {},
                    area_development_score: Math.random(),
                    market_activity_level: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)]
                },
                third_party: {
                    credit_inquiry_activity: Math.random() > 0.5,
                    online_ad_engagement: Math.random(),
                    social_media_signals: { real_estate_engagement: Math.random() }
                },
                consent_given: true
            };
            batchData.push(lead);
        }

        this.currentBatchData = batchData;
        document.getElementById('processBatchBtn').disabled = false;
        
        this.showNotification('Sample batch data generated! Click "Process Batch" to score all leads.', 'success');
    }

    async processBatch() {
        if (!this.currentBatchData) {
            this.showNotification('Please generate sample batch data first.', 'warning');
            return;
        }

        try {
            const batchRequest = {
                leads: this.currentBatchData,
                priority_threshold: 0.7
            };

            const response = await fetch(`${this.apiBase}/score-batch`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(batchRequest)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayBatchResults(result);
        } catch (error) {
            console.error('Batch processing failed:', error);
            this.showNotification('Batch processing failed. Please try again.', 'danger');
        }
    }

    displayBatchResults(result) {
        const batchResults = document.getElementById('batchResults');
        const tableBody = document.getElementById('batchTableBody');
        
        tableBody.innerHTML = '';
        
        result.scores.forEach(score => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${score.lead_id}</td>
                <td><strong>${Math.round(score.final_score)}</strong></td>
                <td><span class="badge priority-${score.priority.toLowerCase()}">${score.priority.toUpperCase()}</span></td>
                <td>${Math.round(score.confidence * 100)}%</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" onclick="app.viewLeadDetails('${score.lead_id}')">
                        <i class="bi bi-eye"></i> View
                    </button>
                </td>
            `;
            tableBody.appendChild(row);
        });

        batchResults.style.display = 'block';
        
        // Show summary notification
        this.showNotification(
            `Processed ${result.total_leads} leads in ${result.processing_time_ms.toFixed(2)}ms. ${result.high_priority_count} high-priority leads found.`,
            'success'
        );
    }

    async loadFeatureImportance() {
        try {
            const response = await fetch(`${this.apiBase}/feature-importance`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displayFeatureImportance(result.feature_importance);
        } catch (error) {
            console.error('Failed to load feature importance:', error);
            this.showNotification('Failed to load feature importance data.', 'danger');
        }
    }

    displayFeatureImportance(importance) {
        const container = document.getElementById('featureImportance');
        
        // Sort features by importance
        const sortedFeatures = Object.entries(importance)
            .sort(([,a], [,b]) => b - a)
            .slice(0, 10); // Top 10 features

        const html = sortedFeatures.map(([feature, score], index) => {
            const percentage = (score * 100).toFixed(1);
            const barWidth = (score / sortedFeatures[0][1] * 100).toFixed(1);
            
            return `
                <div class="mb-2">
                    <div class="d-flex justify-content-between">
                        <small class="fw-bold">${this.formatFeatureName(feature)}</small>
                        <small class="text-muted">${percentage}%</small>
                    </div>
                    <div class="feature-bar">
                        <div class="feature-fill" style="width: ${barWidth}%"></div>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = html;
    }

    formatFeatureName(feature) {
        return feature.split('_').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    async loadSystemMetrics() {
        try {
            const response = await fetch(`${this.apiBase}/metrics`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            this.displaySystemMetrics(result);
        } catch (error) {
            console.error('Failed to load system metrics:', error);
            this.showNotification('Failed to load system metrics.', 'danger');
        }
    }

    displaySystemMetrics(metrics) {
        const health = metrics.model_health;
        
        // Update system health
        const systemHealth = document.getElementById('systemHealth');
        const healthClass = health.status === 'healthy' ? 'health-healthy' : 
                           health.status === 'degraded' ? 'health-degraded' : 'health-unhealthy';
        systemHealth.innerHTML = `
            <span class="health-indicator ${healthClass}"></span>
            <span>${health.status.charAt(0).toUpperCase() + health.status.slice(1)}</span>
        `;

        // Update drift status
        const driftStatus = document.getElementById('driftStatus');
        const hasDrift = health.drift_detection && health.drift_detection.drift_detected;
        driftStatus.innerHTML = hasDrift ? 
            '<span class="text-warning">Drift Detected</span>' : 
            '<span class="text-success">No Drift Detected</span>';

        // Update services status
        const servicesStatus = document.getElementById('servicesStatus');
        servicesStatus.innerHTML = `
            <small>Redis: <span class="text-${health.redis_available ? 'success' : 'warning'}">${health.redis_available ? 'Connected' : 'Fallback'}</span></small><br>
            <small>Model: <span class="text-${health.model_trained ? 'success' : 'warning'}">${health.model_trained ? 'Trained' : 'Not Trained'}</span></small>
        `;

        // Update recent activity
        const recentActivity = document.getElementById('recentActivity');
        const timestamp = new Date(metrics.timestamp).toLocaleString();
        recentActivity.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <small class="text-muted">Last Updated:</small><br>
                    <strong>${timestamp}</strong>
                </div>
                <div class="col-md-6">
                    <small class="text-muted">System Status:</small><br>
                    <strong class="text-success">Operational</strong>
                </div>
            </div>
        `;
    }

    viewLeadDetails(leadId) {
        // In a real app, this would show detailed lead information
        this.showNotification(`Viewing details for ${leadId}`, 'info');
    }

    showNotification(message, type = 'info') {
        // Create and show a temporary notification
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = `
            top: 20px;
            right: 20px;
            z-index: 9999;
            max-width: 400px;
        `;
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new LeadScoringApp();
});

// Standalone functions for HTML onclick handlers
function generateSampleBatch() {
    window.app.generateSampleBatch();
}

function processBatch() {
    window.app.processBatch();
}

function loadFeatureImportance() {
    window.app.loadFeatureImportance();
}

function loadSystemMetrics() {
    window.app.loadSystemMetrics();
}
