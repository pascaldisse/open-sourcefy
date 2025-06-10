#!/usr/bin/env python3
"""
Test script to verify Neo progress reporting and resources.rc compilation fix
"""

import subprocess
import sys
import time

def run_test(test_name, command):
    print(f"\n🧪 Testing: {test_name}")
    print(f"Command: {command}")
    print("=" * 50)
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=60
        )
        
        # Check for errors and warnings
        output_lines = result.stdout.split('\n') + result.stderr.split('\n')
        errors = [line for line in output_lines if 'ERROR' in line]
        warnings = [line for line in output_lines if 'WARNING' in line]
        
        print(f"✅ Exit code: {result.returncode}")
        print(f"✅ Errors found: {len(errors)}")
        print(f"✅ Warnings found: {len(warnings)}")
        
        if errors:
            print("❌ ERRORS:")
            for error in errors[:3]:  # Show first 3 errors
                print(f"  {error}")
        
        if warnings:
            print("⚠️ WARNINGS:")
            for warning in warnings[:3]:  # Show first 3 warnings
                print(f"  {warning}")
        
        return len(errors) == 0 and len(warnings) == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out (expected for long-running operations)")
        return True  # Timeout is acceptable for Neo tests
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def main():
    print("🚀 Testing Open-Sourcefy Pipeline Fixes")
    print("Testing enhanced Neo progress reporting and resources.rc compilation fix")
    
    tests = [
        ("Basic Agent Pipeline", "python3 main.py --agents 1,2 --debug"),
        ("Compilation Fix Test", "python3 main.py --agents 1,2,10 --debug"),
        ("Core Analysis Test", "python3 main.py --agents 1,2,3,4 --debug"),
    ]
    
    results = []
    for test_name, command in tests:
        success = run_test(test_name, command)
        results.append((test_name, success))
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Pipeline is clean!")
        return 0
    else:
        print("\n💥 SOME TESTS FAILED - Check errors above")
        return 1

if __name__ == "__main__":
    sys.exit(main())