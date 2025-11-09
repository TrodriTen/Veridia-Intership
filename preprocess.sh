echo "=========================================="
echo "  Data Preprocessing"
echo "=========================================="
echo ""

if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

if [ ! -f "data/Resume.csv" ]; then
    echo "ERROR: File data/Resume.csv not found"
    echo "Please make sure the data file is in the correct location."
    exit 1
fi

echo "Data file found"
echo ""

echo "Starting data preprocessing..."
echo "This process includes:"
echo "  1. Column selection"
echo "  2. Data split (train/test)"
echo "  3. Text cleaning (tokenization, lemmatization, etc.)"
echo "  4. Class balancing"
echo "  5. Saving processed data"
echo ""

python -m app.preprocessing

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "  Preprocessing completed"
    echo "=========================================="
    echo ""
    echo "Generated files:"
    
    if [ -f "data/train_clean_balanced.csv" ]; then
        echo "  data/train_clean_balanced.csv"
    fi
    
    if [ -f "data/test_clean.csv" ]; then
        echo "  data/test_clean.csv"
    fi
    
    echo ""
    echo "Next step: Train the model"
    echo "  ./train.sh"
else
    echo ""
    echo "=========================================="
    echo "  Preprocessing error"
    echo "=========================================="
    exit 1
fi
