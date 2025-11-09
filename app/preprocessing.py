import re
import unicodedata
import logging

import pandas as pd
import numpy as np
from typing import Tuple
from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK resource: {resource}")
                    nltk.download(resource, quiet=True)
                except:
                    logger.warning(f"Could not download: {resource}")

try:
    download_nltk_resources()
except Exception as e:
    logger.warning(f"Warning downloading NLTK resources: {e}")


def remove_non_ascii(words):
    new_words = []
    for word in words:
        if word is not None:
          new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
          new_words.append(new_word)
    return new_words


def to_lowercase(words):
    new_words = []
    for word in words:
      n1word = word.lower()
      new_words.append(n1word)
    return new_words


def remove_punctuation(words):
    new_words = []
    for word in words:
        if word is not None:
            new_word = re.sub(r'[^\w\s\+\#\.\-\/&]', '', word)
            new_word = re.sub(r'http\S+|www\S+|https\S+', '', new_word, flags=re.MULTILINE)
            new_word = re.sub(r'[^a-zA-Z\s]', ' ', new_word)
            new_word = re.sub(r'\s+', ' ', new_word).strip()
            if new_word != '':
                new_words.append(new_word)
    return new_words


def remove_stopwords(words):
    stop_words = set(stopwords.words('spanish')) | set(stopwords.words('english'))
    new_words = [word for word in words if word not in stop_words]
    return new_words

def lemmatize(words):
    lemmatizer = WordNetLemmatizer()
    new_words = []
    for word in words:
        if len(word) > 2:
            new_word = lemmatizer.lemmatize(word)
            new_words.append(new_word)
    return new_words

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    
    if not text.strip():
        return ""
    
    try:
        tokens = word_tokenize(text)
        
        tokens = remove_punctuation(tokens)
        tokens = to_lowercase(tokens)
        tokens = remove_non_ascii(tokens)
        tokens = remove_stopwords(tokens)
        tokens = lemmatize(tokens)
        
        processed_text = " ".join(tokens)
        
        return processed_text.strip()
    
    except Exception as e:
        logger.error(f"Error preprocessing text: {e}")
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

def select_columns(df: pd.DataFrame, text_col: str = 'Resume_str', 
                   label_col: str = 'Category') -> pd.DataFrame:
    
    logger.info(f"Selecting columns: {text_col}, {label_col}")
    
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Columns not found. Available: {df.columns.tolist()}")
    
    selected = df[[text_col, label_col]].copy()
    selected.columns = ['text', 'label']
    
    logger.info(f"Columns selected. Shape: {selected.shape}")
    return selected


def split_data(df: pd.DataFrame, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:

    logger.info(f"Splitting data: {test_size*100}% for test")
    
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def clean_text_series(X: pd.Series) -> pd.Series:

    logger.info(f"Cleaning {len(X)} texts...")
    X_clean = X.apply(preprocess_text)
    logger.info(f"Cleaning completed")
    return X_clean


def balance_data(X_text: pd.Series, y: pd.Series, 
                 under_cap: int = 100, target_min: int = 80, 
                 random_state: int = 42) -> Tuple[list, list]:

    logger.info("Starting data balancing...")
    logger.info(f"Original distribution:\n{Counter(y)}")
    
    cnt = Counter(y)

    under_strategy = {c: under_cap for c, n in cnt.items() if n > under_cap}

    X_arr = np.array(list(X_text), dtype=object).reshape(-1, 1)

    if under_strategy:
        logger.info(f"Applying under-sampling: {under_strategy}")
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=random_state)
        X_u, y_u = rus.fit_resample(X_arr, y)
    else:
        logger.info("Under-sampling not required")
        X_u, y_u = X_arr, y

    cnt_u = Counter(y_u)
    over_strategy = {c: target_min for c, n in cnt_u.items() if n < target_min}
    
    if over_strategy:
        logger.info(f"Applying over-sampling: {over_strategy}")
        ros = RandomOverSampler(sampling_strategy=over_strategy, random_state=random_state)
        X_b, y_b = ros.fit_resample(X_u, y_u)
    else:
        logger.info("Over-sampling not required")
        X_b, y_b = X_u, y_u

    X_b = X_b.ravel().tolist()
    y_b = list(y_b)
    
    logger.info(f"Balanced distribution:\n{Counter(y_b)}")
    logger.info(f"Total samples: {len(X_b)}")
    
    return X_b, y_b


def save_processed_data(X_train, y_train, X_test, y_test, 
                       train_path: str = "../data/train_clean_balanced.csv",
                       test_path: str = "../data/test_clean.csv"):

    logger.info("Saving processed data...")
    
    df_train = pd.DataFrame({
        "text": pd.Series(X_train, dtype="object"),
        "label": pd.Series(y_train, dtype="object")
    })
    df_train.to_csv(train_path, index=False, encoding="utf-8")
    logger.info(f"Training data saved at: {train_path}")
    
    df_test = pd.DataFrame({
        "text": pd.Series(X_test, dtype="object"),
        "label": pd.Series(y_test, dtype="object")
    })
    df_test.to_csv(test_path, index=False, encoding="utf-8")
    logger.info(f"Test data saved at: {test_path}")


def full_preprocessing_pipeline(input_path: str, 
                                text_col: str = 'Resume_str',
                                label_col: str = 'Category',
                                test_size: float = 0.2,
                                under_cap: int = 100,
                                target_min: int = 80,
                                random_state: int = 42,
                                train_output: str = "../data/train_clean_balanced.csv",
                                test_output: str = "../data/test_clean.csv") -> None:

    logger.info("="*60)
    logger.info("STARTING COMPLETE PREPROCESSING PIPELINE")
    logger.info("="*60)
    
    logger.info(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8', engine='python')
    logger.info(f"Data loaded. Shape: {df.shape}")

    df_selected = select_columns(df, text_col, label_col)

    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df_selected, test_size=test_size, random_state=random_state
    )

    X_train_clean = clean_text_series(X_train_raw)
    X_test_clean = clean_text_series(X_test_raw)

    logger.info("\n--- PREPROCESSING EXAMPLE ---")
    logger.info(f"Original: {X_train_raw.iloc[0][:200]}...")
    logger.info(f"Processed: {X_train_clean.iloc[0][:200]}...")
    
    X_train_bal, y_train_bal = balance_data(
        X_train_clean, y_train, 
        under_cap=under_cap, 
        target_min=target_min, 
        random_state=random_state
    )

    save_processed_data(
        X_train_bal, y_train_bal, 
        X_test_clean, y_test,
        train_output, test_output
    )
    
    logger.info("="*60)
    logger.info("PREPROCESSING COMPLETED SUCCESSFULLY")
    logger.info("="*60)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    base_dir = Path(__file__).resolve().parent.parent
    input_file = str(base_dir / "data" / "Resume.csv")
    train_output = str(base_dir / "data" / "train_clean_balanced.csv")
    test_output = str(base_dir / "data" / "test_clean.csv")
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    full_preprocessing_pipeline(
        input_path=input_file,
        train_output=train_output,
        test_output=test_output
    )
