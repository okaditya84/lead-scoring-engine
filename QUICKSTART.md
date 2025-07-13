# Quick Start Guide - Lead Scoring Engine

## 🚀 Quick Setup (5 minutes)

### Step 1: Setup Environment
```bash
# Run the setup script
setup.bat

# OR manually:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python utils\sample_data_generator.py
```

### Step 2: Configure API Keys (Optional)
1. Copy `.env.example` to `.env`
2. **Optional**: Get your Groq API key from https://console.groq.com/
3. **Optional**: Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

**Note**: The system works perfectly without API keys using ML-only scoring with graceful fallback.

### Step 3: Start the Server
```bash
# Start with Frontend Dashboard (Recommended)
start_frontend.bat

# OR start API only
start.bat
# OR: uvicorn main:app --reload
```

### Step 4: Access the Application
Once the server is running:
- **🎯 Frontend Dashboard**: http://localhost:8000 (Interactive UI)
- **📚 API Documentation**: http://localhost:8000/docs (Swagger UI)
- **🏥 Health Check**: http://localhost:8000/health
- **📊 Metrics**: http://localhost:8000/metrics

### Step 5: Test the System
Open new terminal and run:
```bash
run_tests.bat
# OR: python tests\test_api.py
```

## ⚡ Fallback Features

The Lead Scoring Engine is designed to work seamlessly in development environments:

- **✅ No Redis Required**: All monitoring features have in-memory fallbacks
- **✅ No LLM API Key Required**: Falls back to robust ML-only scoring  
- **✅ No External Dependencies**: Works out-of-the-box after setup
- **✅ Graceful Degradation**: Enhanced features activate when services are available

## 📚 Application Access
Once the server is running:
- **🎯 Frontend Dashboard**: http://localhost:8000 *(NEW! Interactive Web UI)*
- **📚 API Documentation**: http://localhost:8000/docs *(Swagger UI)*
- **🏥 Health Check**: http://localhost:8000/health
- **📊 Metrics**: http://localhost:8000/metrics

## 🎨 Frontend Features

The new web dashboard provides:
- **🖱️ Interactive Lead Scoring**: Visual form-based lead input and scoring
- **📊 Real-time Results**: Instant score visualization with confidence metrics
- **📦 Batch Processing**: Upload and process multiple leads simultaneously
- **📈 Analytics Dashboard**: Feature importance and model performance charts
- **🔍 System Monitoring**: Real-time health status and drift detection
- **📱 Responsive Design**: Works on desktop, tablet, and mobile devices

## 🧪 Quick Test Examples

### Score a Single Lead
```bash
curl -X POST "http://localhost:8000/score-lead" \
  -H "Content-Type: application/json" \
  -d @sample_data/sample_leads.json
```

### Check Model Health
```bash
curl http://localhost:8000/model-health
```

## 🎯 Key Features Demonstrated

✅ **Real-time Scoring**: <300ms latency per lead
✅ **GBT + LLM Architecture**: XGBoost base + Groq LLM re-ranker  
✅ **Multi-source Features**: Behavioral, demographic, public, third-party data
✅ **Automated Notifications**: CRM/WhatsApp integration ready
✅ **Drift Detection**: Automated model performance monitoring
✅ **Continuous Learning**: Automatic retraining pipeline
✅ **DPDP Compliance**: Consent management and data encryption
✅ **Production Ready**: Prometheus metrics, structured logging, health checks

## 📊 Sample Lead Types Included

1. **High Intent Lead**: 
   - Score: 85-95/100
   - Features: High engagement, premium income, credit inquiries
   
2. **Medium Intent Lead**:
   - Score: 50-75/100  
   - Features: Moderate engagement, mixed signals
   
3. **Low Intent Lead**:
   - Score: 15-35/100
   - Features: Low engagement, browsing behavior

## 🔧 Architecture Overview

```
Lead Data → Feature Engineering → XGBoost → LLM Re-ranker → Final Score
    ↓              ↓                ↓          ↓              ↓
Validation → 30+ Features → Base Score → Context Analysis → CRM/WhatsApp
```

## 📈 Monitoring & Metrics

- **Latency Monitoring**: Request/response time tracking
- **Drift Detection**: Feature distribution changes
- **Performance Tracking**: Conversion rate monitoring  
- **Model Health**: Accuracy and confidence metrics

## 🚢 Production Deployment

For production deployment:
1. Use `docker-compose up` for full stack
2. Configure real database connections
3. Set up proper monitoring (Prometheus/Grafana)
4. Configure real CRM/WhatsApp integrations

## 💡 Next Steps

1. **Customize Features**: Modify `features/feature_engineering.py`
2. **Train on Real Data**: Replace demo data with actual lead data
3. **Configure Integrations**: Set up CRM and WhatsApp APIs
4. **Deploy**: Use Docker or cloud platforms
5. **Monitor**: Set up alerts and dashboards

## 🆘 Troubleshooting

**Server won't start?**
- Check if virtual environment is activated
- Verify all dependencies are installed
- Check if port 8000 is available

**Tests failing?** 
- Ensure server is running on localhost:8000
- Check if sample data was generated
- Verify Groq API key is set (for LLM features)

**Low performance?**
- Check system resources
- Monitor /metrics endpoint
- Review logs for errors
