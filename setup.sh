#!/bin/bash
# setup.sh — First-time project setup
set -e

echo "=== Macro Recorder Agent — Setup ==="

# 1. Copy env file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env — EDIT IT before continuing"
    echo "  nano .env"
    exit 0
fi

# 2. Python dependencies
echo "→ Installing Python dependencies..."
pip install -r requirements.txt
playwright install chromium

# 3. Docker infrastructure
echo "→ Starting infrastructure (Docker)..."
docker compose up -d postgres redis minio qdrant prometheus grafana

# 4. Wait for postgres
echo "→ Waiting for PostgreSQL..."
sleep 5

# 5. Run migrations
echo "→ Running database migrations..."
python scripts/migrate.py

# 6. Create MinIO bucket
echo "→ Creating MinIO bucket..."
python scripts/setup_minio.py

# 7. Create Qdrant collection
echo "→ Creating Qdrant collection..."
python scripts/setup_qdrant.py

# 8. Start Langfuse
echo "→ Starting Langfuse..."
docker compose up -d langfuse

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Start the VLM:    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000"
echo "  2. Start agent VM:   docker compose up -d agent-vm"
echo "  3. Test vision:      python scripts/test_vision.py"
echo "  4. Record workflow:  python main.py --record --name test_workflow"
echo "  5. Run agent:        python main.py --task 'Fill the test form'"
echo ""
echo "Dashboards:"
echo "  Agent VM:    http://localhost:6080"
echo "  Langfuse:    http://localhost:3000"
echo "  Grafana:     http://localhost:3001"
echo "  MinIO:       http://localhost:9001"
