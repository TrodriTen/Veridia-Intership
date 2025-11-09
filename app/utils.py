"""
Utility functions for the application
"""
import logging
import io
import PyPDF2
from pathlib import Path


def setup_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def extract_text_from_pdf(file_content: bytes) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def ensure_models_directory(models_dir: Path) -> None:

    models_dir.mkdir(parents=True, exist_ok=True)


def validate_file_size(file_size: int, max_size: int) -> bool:

    return file_size <= max_size


def format_probabilities(probabilities: dict, top_n: int = 5) -> dict:

    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    return dict(list(sorted_probs.items())[:top_n])
