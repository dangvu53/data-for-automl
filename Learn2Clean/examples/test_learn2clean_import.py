#!/usr/bin/env python3
"""
Test script to check if Learn2Clean can be imported correctly
"""

import sys
import os

# Add Learn2Clean to Python path
sys.path.append(os.path.abspath('../python-package'))

try:
    print("Testing Learn2Clean imports...")
    
    # Test basic imports
    import learn2clean
    print("✓ learn2clean imported successfully")
    
    import learn2clean.loading.reader as rd
    print("✓ learn2clean.loading.reader imported successfully")
    
    import learn2clean.normalization.normalizer as nl
    print("✓ learn2clean.normalization.normalizer imported successfully")
    
    import learn2clean.qlearning.qlearner as ql
    print("✓ learn2clean.qlearning.qlearner imported successfully")
    
    print("\n🎉 All Learn2Clean imports successful!")
    print(f"Learn2Clean version: {learn2clean.__version__}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTrying to identify missing dependencies...")
    
    # Check for common missing dependencies
    missing_deps = []
    
    try:
        import pandas
        print("✓ pandas available")
    except ImportError:
        missing_deps.append("pandas")
        
    try:
        import numpy
        print("✓ numpy available")
    except ImportError:
        missing_deps.append("numpy")
        
    try:
        import sklearn
        print("✓ sklearn available")
    except ImportError:
        missing_deps.append("sklearn")
        
    if missing_deps:
        print(f"\n❌ Missing dependencies: {missing_deps}")
        print("Please install them with: pip install " + " ".join(missing_deps))
    else:
        print("\n🤔 All basic dependencies seem available. The issue might be with Learn2Clean's internal dependencies.")
        
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
