#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Initialize the model with demo data
python -c "
import asyncio
from main import _initialize_demo_model
asyncio.run(_initialize_demo_model())
"

echo "Build completed successfully!"
