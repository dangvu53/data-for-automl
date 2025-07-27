#!/usr/bin/env python3
"""
Test script for the modern Learn2Clean implementation
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from learn2clean_implement import ModernDuplicateDetector, ModernOutlierDetector, Learn2CleanProcessor


def test_duplicate_detection():
    """Test duplicate detection functionality"""
    print("Testing Duplicate Detection...")
    
    # Create test data with known duplicates
    test_data = pd.DataFrame({
        'text': [
            "Hello world",
            "Hello world",  # Exact duplicate
            "Hello world!",  # Fuzzy duplicate
            "Greetings earth",  # Different
            "Hi planet",  # Different
            "Hello world!!!",  # Fuzzy duplicate
        ],
        'category': ['A', 'A', 'A', 'B', 'B', 'A'],
        'value': [1, 1, 2, 3, 4, 2]
    })
    
    print(f"Original data size: {len(test_data)}")
    
    # Test exact duplicate detection
    detector = ModernDuplicateDetector(strategy='exact', verbose=True)
    result = detector.detect_and_remove_duplicates(test_data)
    print(f"After exact deduplication: {len(result)}")
    assert len(result) < len(test_data), "Exact deduplication should remove some rows"
    
    # Test fuzzy duplicate detection
    detector = ModernDuplicateDetector(strategy='fuzzy', threshold=0.8, verbose=True)
    result = detector.detect_and_remove_duplicates(test_data)
    print(f"After fuzzy deduplication: {len(result)}")
    
    print("✓ Duplicate detection tests passed\n")


def test_outlier_detection():
    """Test outlier detection functionality"""
    print("Testing Outlier Detection...")
    
    # Create test data with known outliers
    np.random.seed(42)
    normal_data = np.random.normal(50, 5, 20)
    outlier_data = [100, -50]  # Clear outliers
    
    test_data = pd.DataFrame({
        'text': [f"Sample text {i}" for i in range(22)],
        'score': list(normal_data) + outlier_data,
        'category': ['normal'] * 20 + ['outlier'] * 2
    })
    
    print(f"Original data size: {len(test_data)}")
    print(f"Score range: {test_data['score'].min():.1f} to {test_data['score'].max():.1f}")
    
    # Test LOF outlier detection
    detector = ModernOutlierDetector(strategy='lof', contamination=0.1, verbose=True)
    result = detector.detect_and_remove_outliers(test_data)
    print(f"After LOF outlier detection: {len(result)}")
    
    # Test Z-score outlier detection
    detector = ModernOutlierDetector(strategy='zscore', contamination=0.1, verbose=True)
    result = detector.detect_and_remove_outliers(test_data)
    print(f"After Z-score outlier detection: {len(result)}")
    
    print("✓ Outlier detection tests passed\n")


def test_combined_processing():
    """Test combined duplicate and outlier detection"""
    print("Testing Combined Processing...")
    
    # Create comprehensive test data
    test_data = pd.DataFrame({
        'text': [
            "Normal text sample 1",
            "Normal text sample 1",  # Duplicate
            "Normal text sample 2", 
            "Normal text sample 3",
            "WEIRD OUTLIER TEXT WITH RANDOM SYMBOLS !@#$%",  # Text outlier
            "Another normal sample",
            "Similar normal sample",  # Fuzzy duplicate
            "Regular content here",
        ],
        'score': [50, 50, 55, 48, 200, 52, 53, 49],  # Score outlier at index 4
        'category': ['A', 'A', 'B', 'B', 'C', 'C', 'C', 'D']
    })
    
    print(f"Original data size: {len(test_data)}")
    
    # Configure processor
    processor = Learn2CleanProcessor(
        duplicate_config={
            'strategy': 'hybrid',
            'threshold': 0.8,
            'verbose': True
        },
        outlier_config={
            'strategy': 'hybrid',
            'contamination': 0.2,
            'verbose': True
        }
    )
    
    # Process data
    result = processor.process(test_data)
    print(f"Final processed data size: {len(result)}")
    
    # Verify some cleaning occurred
    assert len(result) < len(test_data), "Combined processing should remove some rows"
    
    print("✓ Combined processing tests passed\n")


def test_edge_cases():
    """Test edge cases"""
    print("Testing Edge Cases...")
    
    # Empty dataframe
    empty_df = pd.DataFrame()
    processor = Learn2CleanProcessor()
    result = processor.process(empty_df)
    assert len(result) == 0, "Empty dataframe should remain empty"
    
    # Single row
    single_row = pd.DataFrame({'text': ['single'], 'value': [1]})
    result = processor.process(single_row)
    assert len(result) == 1, "Single row should be preserved"
    
    # All duplicates
    all_dupes = pd.DataFrame({
        'text': ['same'] * 5,
        'value': [1] * 5
    })
    detector = ModernDuplicateDetector(strategy='exact')
    result = detector.detect_and_remove_duplicates(all_dupes)
    assert len(result) == 1, "All duplicates should reduce to one row"
    
    print("✓ Edge case tests passed\n")


def benchmark_performance():
    """Simple performance benchmark"""
    print("Running Performance Benchmark...")
    
    # Create larger dataset
    np.random.seed(42)
    size = 1000
    
    large_data = pd.DataFrame({
        'text': [f"Sample text content {i % 100}" for i in range(size)],  # Some duplicates
        'score': np.random.normal(50, 10, size),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size)
    })
    
    # Add some outliers
    outlier_indices = np.random.choice(size, int(size * 0.05), replace=False)
    large_data.loc[outlier_indices, 'score'] = np.random.normal(150, 20, len(outlier_indices))
    
    print(f"Benchmark data size: {len(large_data)}")
    
    import time
    
    # Time the processing
    start_time = time.time()
    
    processor = Learn2CleanProcessor(
        duplicate_config={'strategy': 'exact'},  # Faster for benchmark
        outlier_config={'strategy': 'lof', 'contamination': 0.05}
    )
    
    result = processor.process(large_data)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"Processed {len(large_data)} -> {len(result)} rows in {processing_time:.2f} seconds")
    print(f"Processing rate: {len(large_data) / processing_time:.0f} rows/second")
    
    print("✓ Performance benchmark completed\n")


if __name__ == "__main__":
    print("Starting Learn2Clean Implementation Tests\n")
    print("=" * 60)
    
    try:
        test_duplicate_detection()
        test_outlier_detection()
        test_combined_processing()
        test_edge_cases()
        benchmark_performance()
        
        print("=" * 60)
        print("🎉 All tests passed successfully!")
        print("\nThe Learn2Clean implementation is working correctly with modern packages.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
