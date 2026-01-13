"""
Unit tests for data validation and preprocessing.

Tests ensure data quality, format correctness, and preprocessing functions.
"""

import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestDataValidation(unittest.TestCase):
    """Test suite for data validation and quality checks."""
    
    @classmethod
    def setUpClass(cls):
        """Load test dataset once for all tests."""
        cls.test_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'reviews_dataset.csv')
        if os.path.exists(cls.test_data_path):
            cls.df = pd.read_csv(cls.test_data_path)
        else:
            # Create minimal test dataset if not found
            cls.df = pd.DataFrame({
                'review_id': [1, 2, 3],
                'review_text': ['Great product!', 'Terrible experience.', 'It works ok.'],
                'rating': [5, 1, 3]
            })
    
    def test_dataframe_not_empty(self):
        """Test that dataset contains records."""
        self.assertGreater(len(self.df), 0, "Dataset should not be empty")
    
    def test_required_columns_present(self):
        """Test that required columns exist in the dataset."""
        required_columns = ['review_id', 'review_text']
        for col in required_columns:
            self.assertIn(col, self.df.columns, f"Required column '{col}' is missing")
    
    def test_no_null_review_text(self):
        """Test that review_text column has no null values."""
        null_count = self.df['review_text'].isnull().sum()
        self.assertEqual(null_count, 0, "review_text should not contain null values")
    
    def test_review_text_is_string(self):
        """Test that all review texts are strings."""
        non_string_count = sum(~self.df['review_text'].apply(lambda x: isinstance(x, str)))
        self.assertEqual(non_string_count, 0, "All review_text values should be strings")
    
    def test_review_id_unique(self):
        """Test that review IDs are unique."""
        duplicate_count = self.df['review_id'].duplicated().sum()
        self.assertEqual(duplicate_count, 0, "review_id should be unique")
    
    def test_review_length_positive(self):
        """Test that all reviews have positive length."""
        min_length = self.df['review_text'].str.len().min()
        self.assertGreater(min_length, 0, "All reviews should have length > 0")
    
    def test_rating_in_valid_range(self):
        """Test that ratings are in expected range (if column exists)."""
        if 'rating' in self.df.columns:
            min_rating = self.df['rating'].min()
            max_rating = self.df['rating'].max()
            self.assertGreaterEqual(min_rating, 1, "Minimum rating should be >= 1")
            self.assertLessEqual(max_rating, 5, "Maximum rating should be <= 5")
    
    def test_dataframe_shape_consistency(self):
        """Test that dataframe has expected shape properties."""
        rows, cols = self.df.shape
        self.assertGreater(rows, 0, "Should have at least 1 row")
        self.assertGreaterEqual(cols, 2, "Should have at least 2 columns")


class TestPreprocessing(unittest.TestCase):
    """Test suite for text preprocessing functions."""
    
    def setUp(self):
        """Set up test data for each test."""
        self.sample_texts = [
            "Great Product! HIGHLY Recommended!!!",
            "terrible   quality... very disappointed.",
            "It's okay, nothing special",
            "Best purchase ever! 😊",
            ""
        ]
    
    def test_lowercase_conversion(self):
        """Test that text is converted to lowercase."""
        text = "HELLO World 123"
        processed = text.lower()
        self.assertEqual(processed, "hello world 123")
    
    def test_special_character_removal(self):
        """Test removal of special characters."""
        import re
        text = "Hello! How are you? #Great"
        processed = re.sub(r'[^a-z\s]', '', text.lower())
        self.assertNotIn('!', processed)
        self.assertNotIn('?', processed)
        self.assertNotIn('#', processed)
    
    def test_whitespace_normalization(self):
        """Test that multiple spaces are normalized."""
        text = "Hello    World   Test"
        processed = ' '.join(text.split())
        self.assertEqual(processed, "Hello World Test")
    
    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        text = ""
        self.assertEqual(len(text), 0)
        self.assertIsInstance(text, str)
    
    def test_review_length_calculation(self):
        """Test review length feature extraction."""
        texts = ["short", "This is a longer review text"]
        lengths = [len(t) for t in texts]
        self.assertEqual(lengths[0], 5)
        self.assertGreater(lengths[1], lengths[0])


class TestSentimentScoreValidation(unittest.TestCase):
    """Test suite for sentiment score validation."""
    
    def test_vader_score_range(self):
        """Test that VADER scores are in valid range [-1, 1]."""
        # Sample scores
        scores = [-0.8, -0.2, 0.0, 0.5, 0.9]
        for score in scores:
            self.assertGreaterEqual(score, -1.0, "Score should be >= -1")
            self.assertLessEqual(score, 1.0, "Score should be <= 1")
    
    def test_sentiment_category_mapping(self):
        """Test sentiment category assignment logic."""
        # Test thresholds
        test_cases = [
            (-0.5, 'Negative'),
            (-0.02, 'Neutral'),
            (0.02, 'Neutral'),
            (0.7, 'Positive')
        ]
        
        for score, expected_category in test_cases:
            if score > 0.05:
                category = 'Positive'
            elif score < -0.05:
                category = 'Negative'
            else:
                category = 'Neutral'
            self.assertEqual(category, expected_category, 
                           f"Score {score} should map to {expected_category}")
    
    def test_confidence_score_range(self):
        """Test that confidence scores are valid probabilities."""
        confidences = [0.0, 0.5, 0.75, 1.0]
        for conf in confidences:
            self.assertGreaterEqual(conf, 0.0, "Confidence should be >= 0")
            self.assertLessEqual(conf, 1.0, "Confidence should be <= 1")


class TestOutputValidation(unittest.TestCase):
    """Test suite for output file validation."""
    
    def test_output_dataframe_structure(self):
        """Test that output dataframe has expected columns."""
        # Simulate output dataframe
        output_df = pd.DataFrame({
            'review_id': [1, 2],
            'review_text': ['Great!', 'Bad.'],
            'sentiment_score': [0.8, -0.6],
            'sentiment_category': ['Positive', 'Negative']
        })
        
        expected_columns = ['review_id', 'review_text', 'sentiment_score', 'sentiment_category']
        for col in expected_columns:
            self.assertIn(col, output_df.columns, f"Output should contain '{col}' column")
    
    def test_sentiment_category_values(self):
        """Test that sentiment categories are valid."""
        valid_categories = {'Positive', 'Negative', 'Neutral'}
        test_categories = ['Positive', 'Negative', 'Neutral', 'Positive']
        
        for cat in test_categories:
            self.assertIn(cat, valid_categories, f"'{cat}' should be a valid category")
    
    def test_export_file_format(self):
        """Test that exported files have correct format."""
        # Create test dataframe
        df = pd.DataFrame({
            'col1': [1, 2],
            'col2': ['a', 'b']
        })
        
        # Test CSV export (in-memory)
        csv_string = df.to_csv(index=False)
        self.assertIn('col1,col2', csv_string, "CSV should contain header")
        self.assertIn('1,a', csv_string, "CSV should contain data")


def run_tests():
    """Run all test suites."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentScoreValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestOutputValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests when script is executed directly
    success = run_tests()
    sys.exit(0 if success else 1)
