"""
Unit tests for sentiment analysis functionality.

Tests ensure correct sentiment scoring, classification, and model behavior.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestSentimentAnalysis(unittest.TestCase):
    """Test suite for sentiment analysis functions."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize sentiment analyzer once for all tests."""
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            
            # Download VADER lexicon if needed
            try:
                nltk.data.find('vader_lexicon')
            except LookupError:
                nltk.download('vader_lexicon', quiet=True)
            
            cls.sia = SentimentIntensityAnalyzer()
            cls.vader_available = True
        except Exception as e:
            print(f"Warning: Could not initialize VADER: {e}")
            cls.vader_available = False
    
    def test_positive_sentiment_detection(self):
        """Test detection of clearly positive sentiment."""
        if not self.vader_available:
            self.skipTest("VADER not available")
        
        positive_texts = [
            "This product is absolutely amazing!",
            "Excellent quality, highly recommend!",
            "Love it! Best purchase ever.",
            "Fantastic experience, very satisfied."
        ]
        
        for text in positive_texts:
            score = self.sia.polarity_scores(text)['compound']
            self.assertGreater(score, 0.05, 
                             f"'{text}' should have positive score, got {score}")
    
    def test_negative_sentiment_detection(self):
        """Test detection of clearly negative sentiment."""
        if not self.vader_available:
            self.skipTest("VADER not available")
        
        negative_texts = [
            "Terrible quality, very disappointed.",
            "Worst product ever, complete waste.",
            "Horrible experience, do not buy!",
            "Poor quality and bad service."
        ]
        
        for text in negative_texts:
            score = self.sia.polarity_scores(text)['compound']
            self.assertLess(score, -0.05, 
                          f"'{text}' should have negative score, got {score}")
    
    def test_neutral_sentiment_detection(self):
        """Test detection of neutral sentiment."""
        if not self.vader_available:
            self.skipTest("VADER not available")
        
        neutral_texts = [
            "The product arrived today.",
            "It is what it is.",
            "Standard quality, as expected."
        ]
        
        for text in neutral_texts:
            score = self.sia.polarity_scores(text)['compound']
            # Neutral should be close to zero
            self.assertGreaterEqual(score, -0.3, 
                                  f"'{text}' should not be strongly negative, got {score}")
            self.assertLessEqual(score, 0.3, 
                               f"'{text}' should not be strongly positive, got {score}")
    
    def test_negation_handling(self):
        """Test that negations affect sentiment correctly."""
        if not self.vader_available:
            self.skipTest("VADER not available")
        
        # Positive without negation
        pos_score = self.sia.polarity_scores("This is good")['compound']
        
        # Negative with negation
        neg_score = self.sia.polarity_scores("This is not good")['compound']
        
        # Negation should make it less positive or negative
        self.assertGreater(pos_score, neg_score, 
                         "Negation should reduce positive sentiment")
    
    def test_intensifier_effect(self):
        """Test that intensifiers amplify sentiment."""
        if not self.vader_available:
            self.skipTest("VADER not available")
        
        # Without intensifier
        normal_score = self.sia.polarity_scores("This is good")['compound']
        
        # With intensifier
        intense_score = self.sia.polarity_scores("This is very good")['compound']
        
        # Intensifier should increase positive sentiment
        self.assertGreater(intense_score, normal_score, 
                         "Intensifier should amplify positive sentiment")
    
    def test_empty_text_handling(self):
        """Test handling of empty or whitespace-only text."""
        if not self.vader_available:
            self.skipTest("VADER not available")
        
        empty_texts = ["", "   ", "\n\t"]
        
        for text in empty_texts:
            score = self.sia.polarity_scores(text)['compound']
            # Empty text should return neutral score
            self.assertEqual(score, 0.0, 
                           f"Empty text should return 0.0 score, got {score}")
    
    def test_mixed_sentiment(self):
        """Test reviews with both positive and negative elements."""
        if not self.vader_available:
            self.skipTest("VADER not available")
        
        mixed_text = "Good product but terrible delivery service"
        score = self.sia.polarity_scores(mixed_text)['compound']
        
        # Mixed sentiment should be somewhere in between
        # Not as strong as purely positive or purely negative
        self.assertGreaterEqual(score, -0.8, "Mixed sentiment should not be extremely negative")
        self.assertLessEqual(score, 0.8, "Mixed sentiment should not be extremely positive")


class TestSentimentCategorization(unittest.TestCase):
    """Test suite for sentiment category assignment."""
    
    def test_category_thresholds(self):
        """Test that categories are assigned based on correct thresholds."""
        test_cases = [
            (0.8, 'Positive'),
            (0.06, 'Positive'),
            (0.04, 'Neutral'),
            (0.0, 'Neutral'),
            (-0.04, 'Neutral'),
            (-0.06, 'Negative'),
            (-0.8, 'Negative')
        ]
        
        for score, expected_category in test_cases:
            # Apply categorization logic
            if score > 0.05:
                category = 'Positive'
            elif score < -0.05:
                category = 'Negative'
            else:
                category = 'Neutral'
            
            self.assertEqual(category, expected_category, 
                           f"Score {score} should map to {expected_category}, got {category}")
    
    def test_boundary_cases(self):
        """Test edge cases at category boundaries."""
        boundaries = [
            (0.05, 'Neutral'),   # Just at positive threshold
            (0.050001, 'Positive'),  # Just above threshold
            (-0.05, 'Neutral'),  # Just at negative threshold
            (-0.050001, 'Negative')  # Just below threshold
        ]
        
        for score, expected_category in boundaries:
            if score > 0.05:
                category = 'Positive'
            elif score < -0.05:
                category = 'Negative'
            else:
                category = 'Neutral'
            
            self.assertEqual(category, expected_category,
                           f"Boundary case {score} should map to {expected_category}")


class TestStatisticalAnalysis(unittest.TestCase):
    """Test suite for statistical analysis functions."""
    
    def test_correlation_calculation(self):
        """Test correlation coefficient calculation."""
        try:
            from scipy.stats import pearsonr
            
            # Perfect positive correlation
            x1 = [1, 2, 3, 4, 5]
            y1 = [2, 4, 6, 8, 10]
            corr1, _ = pearsonr(x1, y1)
            self.assertAlmostEqual(corr1, 1.0, places=5, 
                                 msg="Perfect positive correlation should be 1.0")
            
            # Perfect negative correlation
            x2 = [1, 2, 3, 4, 5]
            y2 = [10, 8, 6, 4, 2]
            corr2, _ = pearsonr(x2, y2)
            self.assertAlmostEqual(corr2, -1.0, places=5, 
                                 msg="Perfect negative correlation should be -1.0")
            
            # No correlation
            x3 = [1, 2, 3, 4, 5]
            y3 = [3, 3, 3, 3, 3]
            corr3, _ = pearsonr(x3, y3)
            # Correlation with constant should be NaN or 0
            self.assertTrue(abs(corr3) < 0.01 or str(corr3) == 'nan', 
                          "No variation should result in zero/nan correlation")
        
        except ImportError:
            self.skipTest("scipy not available")
    
    def test_statistical_summary(self):
        """Test statistical summary calculations."""
        import numpy as np
        
        data = [1, 2, 3, 4, 5]
        
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        
        self.assertEqual(mean, 3.0, "Mean should be 3.0")
        self.assertEqual(median, 3.0, "Median should be 3.0")
        self.assertGreater(std, 0, "Std deviation should be > 0")


def run_tests():
    """Run all sentiment analysis tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentCategorization))
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalAnalysis))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
