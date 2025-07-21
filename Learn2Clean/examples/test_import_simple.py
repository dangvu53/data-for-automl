#!/usr/bin/env python3

import sys
import os
sys.path.append('../python-package')

try:
    print("Testing Learn2Clean import...")
    import learn2clean.loading.reader as rd
    print("✅ SUCCESS: Learn2Clean imported successfully!")
    
    # Test basic functionality
    print("Testing basic functionality...")
    print("Available functions in reader:", [x for x in dir(rd) if not x.startswith('_')])
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
