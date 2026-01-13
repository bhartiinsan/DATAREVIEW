"""
Test runner for all test suites.

Run all tests with: python -m tests.run_all_tests
Or from project root: python -m pytest tests/
"""

import sys
import os
import unittest

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tests.test_data_validation import TestDataValidation, TestPreprocessing, TestSentimentScoreValidation, TestOutputValidation
from tests.test_sentiment_analysis import TestSentimentAnalysis, TestSentimentCategorization, TestStatisticalAnalysis


def main():
    """Run all test suites and report results."""
    
    print("="*70)
    print("RUNNING ALL TESTS - DATAREVIEW Sentiment Analysis")
    print("="*70)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    print("Loading test suites...")
    test_classes = [
        TestDataValidation,
        TestPreprocessing,
        TestSentimentScoreValidation,
        TestOutputValidation,
        TestSentimentAnalysis,
        TestSentimentCategorization,
        TestStatisticalAnalysis
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        print(f"  [OK] Loaded {test_class.__name__} ({tests.countTestCases()} tests)")
    
    print(f"\nTotal tests to run: {suite.countTestCases()}")
    print("="*70)
    print()
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    # Return exit code
    if result.wasSuccessful():
        print("\n[PASS] ALL TESTS PASSED!")
        return 0
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
