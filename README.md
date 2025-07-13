# Lead Scoring Engine

A comprehensive real estate lead scoring system using Gradient Boosted Trees and LLM re-ranking for high-intent prospect prediction.

## Features

- **Real-time Scoring**: <300ms latency for lead scoring
- **Multi-source Data**: Behavioral, demographic, public, and third-party data integration
- **GBT + LLM Architecture**: XGBoost base model with Groq LLM re-ranker
- **CRM Integration**: Direct integration with CRM and WhatsApp systems
- **Monitoring & Drift Detection**: Automated model performance tracking
- **Compliance Ready**: DPDP Act compliant with consent management

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Run the application:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. Access the API documentation:
```
http://localhost:8000/docs
```

## Architecture

```
Data Sources → Feature Store → GBT Model → LLM Re-ranker → CRM/WhatsApp
     ↓              ↓            ↓           ↓              ↓
  Kafka Stream → Redis Cache → XGBoost → Groq API → REST APIs
```

## API Endpoints

- `POST /score-lead`: Score a single lead
- `POST /score-batch`: Score multiple leads
- `GET /model-health`: Check model status
- `GET /metrics`: Prometheus metrics
- `POST /retrain`: Trigger model retraining

## Monitoring

- Feature drift detection
- Model performance tracking
- Real-time latency monitoring
- Conversion feedback loop

## License

MIT License
