#!/bin/bash

# Test the image generation API with curl

curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Design a futuristic poster for iPhone 16 Black Friday Sale 50% OFF, dark neon background, golden glowing text, sleek product highlight.",
    "logo_path": "./logo.png",
    "phone_number": "0909 123 456"
  }'

echo ""
