echo "=========================================="
echo "  Starting Resume Classifier API"
echo "=========================================="
echo ""

if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

if [ ! -f "app/models/model.pkl" ] || [ ! -f "app/models/vectorizer.pkl" ]; then
    echo "WARNING: Models not found"
    echo "Please run first: ./train.sh"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting FastAPI server..."
echo "API available at: http://localhost:8000"
echo "Documentation at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
