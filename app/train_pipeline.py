import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

from app.config import (
    TRAIN_DATA_PATH, 
    TEST_DATA_PATH,
    VECTORIZER_PATH, 
    MODEL_PATH,
    RANDOM_STATE,
    NGRAM_RANGE,
    LOWERCASE,
)
from app.utils import ensure_models_directory, setup_logging

logger = logging.getLogger(__name__)


class ResumeClassifierPipeline:
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        
    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()

        if not {'text', 'label'}.issubset(df.columns):
            missing = {'text', 'label'} - set(df.columns)
            raise ValueError(f'Missing columns: {missing}')

        df = df.dropna(subset=['label'])

        df['text'] = df['text'].fillna('').astype(str).str.strip()
        df = df[df['text'].str.len() > 0]
        
        logger.info(f"DataFrame sanitized: {len(df)} valid records")
        return df
    
    def load_data(self, train_path: Optional[Path] = None, 
                  test_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        train_path = train_path or TRAIN_DATA_PATH
        test_path = test_path or TEST_DATA_PATH
        
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path, encoding='utf-8')
        train_df = self._sanitize_dataframe(train_df)
        
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path, encoding='utf-8')
        test_df = self._sanitize_dataframe(test_df)
        
        return train_df, test_df
    
    def vectorize(self, X_train: pd.Series, X_test: pd.Series):

        logger.info("Creating CounterVectorizer...")
        self.vectorizer = CountVectorizer(
            ngram_range=NGRAM_RANGE,
            lowercase=LOWERCASE,
            max_features=10000
        )
        
        logger.info("Vectorizing training data...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        logger.info(f"Training shape: {X_train_vec.shape}")
        
        logger.info("Vectorizing test data...")
        X_test_vec = self.vectorizer.transform(X_test)
        logger.info(f"Test shape: {X_test_vec.shape}")
        
        return X_train_vec, X_test_vec
    
    def train(self, X_train_vec, y_train):

        logger.info("Starting training...")
        
        self.model = RandomForestClassifier(
            random_state=RANDOM_STATE,
            max_features=0.5,
            criterion='log_loss'
        )
        
        self.model.fit(X_train_vec, y_train)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Model trained successfully")
        logger.info(f"{'='*60}\n")
        
    def evaluate(self, X_test_vec, y_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        logger.info("Evaluating model on test set...")
        y_pred = self.model.predict(X_test_vec)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST RESULTS:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1-macro: {f1:.4f}")
        logger.info(f"{'='*60}\n")
        logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
        
    def save_models(self, vectorizer_path: Optional[Path] = None, 
                    model_path: Optional[Path] = None):
        vectorizer_path = vectorizer_path or VECTORIZER_PATH
        model_path = model_path or MODEL_PATH
        
        ensure_models_directory(vectorizer_path.parent)
        
        logger.info(f"Saving vectorizer at {vectorizer_path}")
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        logger.info(f"Saving model at {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        logger.info("Models saved successfully")
    
    def run_full_pipeline(self):

        logger.info("="*60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*60)

        train_df, test_df = self.load_data()
        X_train, y_train = train_df['text'], train_df['label']
        X_test, y_test = test_df['text'], test_df['label']
        
        logger.info(f"Unique classes: {sorted(y_train.unique())}")
        logger.info(f"Training distribution:\n{y_train.value_counts()}")
        
        X_train_vec, X_test_vec = self.vectorize(X_train, X_test)

        self.train(X_train_vec, y_train)

        self.evaluate(X_test_vec, y_test)

        self.save_models()
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)


def main():
    """Main function to execute the pipeline"""
    setup_logging("INFO")
    
    pipeline = ResumeClassifierPipeline()
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
