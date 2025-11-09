# Resume Classifier 

Resume classification system using Machine Learning with FastAPI and web interface.

## Features

- Automatic resume classification into 23+ professional categories
- Modern web interface for easy testing
- REST API with FastAPI
- Support for TXT and PDF files
- Text preprocessing with NLTK
- Optimized Random Forest model
- Interactive API documentation with Swagger

## Project Structure

```
Veridia/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── train_pipeline.py       # Training pipeline
│   ├── preprocessing.py        # Text preprocessing
│   ├── schemas.py             # Pydantic models
│   ├── config.py              # Configuration
│   ├── utils.py               # Utility functions
│   └── models/                # Trained models (.pkl)
├── frontend/
│   ├── index.html             # Web interface
│   ├── styles.css             # Styles
│   └── script.js              # Frontend logic
├── data/
│   ├── Resume.csv             # Original dataset
│   ├── train_clean_balanced.csv
│   └── test_clean.csv
├── jupyter/                    # Development notebooks
├── requirements.txt
├── preprocess.sh              # Script to preprocess data
├── train.sh                   # Script to train model
└── run.sh                      # Script to run API

```

## Quick Start

### 1. Preprocess Data (if needed)

```bash
./preprocess.sh
```

### 2. Train the Model (first time)

```bash
./train.sh
# or
python -m app.train_pipeline
```

### 3. Run the Application

```bash
./run.sh
# or
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access the application:
- **Web Interface**: http://localhost:8000 (Main interface for testing)
- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Usage

### Web Interface (Recommended)

1. Run the application with `./run.sh`
2. Open http://localhost:8000 in your browser
3. Choose between two options:
   - **Text Input**: Paste resume text directly
   - **Upload File**: Upload a TXT or PDF file
4. Click "Analyze Resume" to get results
5. View the predicted category, confidence, and top probabilities

The web interface provides:
- Visual and intuitive results
- Confidence charts and probabilities
- Drag and drop file upload
- Real-time validation
- Modern and responsive design

### API Usage (Programmatic)

#### 1. Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Predict from text
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Software Engineer with 5 years of experience in Python, Django, FastAPI. Developed REST APIs and microservices..."
  }'
```

#### 3. Predict from a file
```bash
curl -X POST "http://localhost:8000/predict/file" \
  -F "file=@resume.pdf"
```

#### 4. Get Available Categories
```bash
curl http://localhost:8000/categories
```

### Example Response

```json
{
  "category": "Data Science",
  "confidence": 0.89,
  "probabilities": {
    "Data Science": 0.89,
    "Web Developing": 0.05,
    "Python Developer": 0.03,
    "Other": 0.03
  }
}
```

## Dependencies

Main dependencies:

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **scikit-learn**: Machine Learning
- **NLTK**: Natural language processing
- **pandas/numpy**: Data manipulation
- **PyPDF2**: PDF reading

Full installation:

```bash
pip install -r requirements.txt
```

## Configuration

Configuration is located in `app/config.py`:

- Model and data paths
- Model parameters
- API configuration
- Validations

## Model

- **Algorithm**: Random Forest Classifier
- **Vectorization**: Count Vectorizer with n-grams (1,2)
- **Balancing**: Under-sampling + Over-sampling
- **Features**: 10,000 maximum features
- **Evaluation**: F1-macro and Accuracy

## Testing

### Option 1: Web Interface (Easiest)

1. Run `./run.sh`
2. Open http://localhost:8000
3. Test with resume text or upload files
4. See visual results immediately

### Option 2: Interactive API Documentation

1. Open http://localhost:8000/docs
2. Test endpoints directly from the browser
3. See examples and data schemas

### Option 3: Command Line

Use curl commands as shown in the API Usage section above.

## Troubleshooting

### Models not found
```bash
./train.sh
```

### Missing dependencies
```bash
pip install -r requirements.txt
```

### Training data not found
```bash
./preprocess.sh
```

### Script permission errors
```bash
chmod +x preprocess.sh train.sh run.sh
```

## Endpoint Documentation

### GET `/`
Web interface - Interactive UI for resume classification

### GET `/health`
API and model status

### POST `/predict`
Prediction from JSON text

**Request Body:**
```json
{
  "text": "Your resume text here..."
}
```

### POST `/predict/file`
Prediction from file (TXT or PDF, max 10MB)

**Form Data:**
- `file`: Resume file

### GET `/categories`
List of 23+ available professional categories

## Contributing

This is an internship project. For improvements:
1. Review notebooks in `jupyter/`
2. Modify code in `app/`
3. Retrain the model
4. Test with the API

## License

Practice project - Veridia Internship

