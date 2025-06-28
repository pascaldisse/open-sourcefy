#!/usr/bin/env python3
"""
Debug script to achieve 100% functional identity and verify Matrix EULA replacement
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('launcher_debug.log'),
        logging.StreamHandler()
    ]
)

def run_matrix_pipeline():
    """Run Matrix pipeline until 100% binary match is achieved"""
    logging.info("🚀 MISSION: Achieve 100% functional identity")
    
    attempts = 0
    max_attempts = 10
    
    while attempts < max_attempts:
        attempts += 1
        logging.info(f"🔄 Binary reconstruction attempt {attempts}/{max_attempts}")
        
        # Run Matrix pipeline
        result = subprocess.run([
            sys.executable, 'main.py'
        ], capture_output=True, text=True, cwd='/mnt/c/Users/pascaldisse/Downloads/open-sourcefy')
        
        if result.returncode == 0:
            logging.info("✅ Matrix pipeline completed successfully")
            
            # Check validation results
            validation_file = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/reports/final_validation_report.json'
            if os.path.exists(validation_file):
                import json
                with open(validation_file, 'r') as f:
                    validation = json.load(f)
                
                total_match = validation.get('final_validation', {}).get('total_match_percentage', 0)
                binary_comparison = validation.get('binary_comparison', {})
                size_match = binary_comparison.get('size_match', 0)
                hash_match = binary_comparison.get('hash_match', 0)
                
                logging.info(f"📊 Total match: {total_match}%")
                logging.info(f"📊 Size match: {size_match}%")
                logging.info(f"📊 Hash match: {hash_match}%")
                
                if total_match >= 100.0 and size_match >= 100.0 and hash_match >= 100.0:
                    logging.info("🎉 100% FUNCTIONAL IDENTITY ACHIEVED!")
                    return True
                else:
                    logging.warning(f"❌ Binary reconstruction incomplete - continuing attempts")
            
        else:
            logging.error(f"❌ Pipeline failed: {result.stderr}")
        
        # Apply Matrix EULA fix after each attempt
        apply_matrix_eula_fix()
        
    logging.error("❌ Failed to achieve 100% functional identity after maximum attempts")
    return False

def apply_matrix_eula_fix():
    """Apply Matrix EULA replacement to binary resources"""
    logging.info("🔧 Applying Matrix EULA replacement")
    
    resources_file = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/compilation/raw_resources.res'
    
    if os.path.exists(resources_file):
        try:
            with open(resources_file, 'rb') as f:
                data = bytearray(f.read())
            
            # Replace EULA text
            old_text = 'End User License Agreement'
            new_text = 'Matrix Digital Agreement'
            old_bytes = old_text.encode('utf-16le')
            new_bytes = new_text.encode('utf-16le')
            
            pos = data.find(old_bytes)
            if pos > -1:
                padding = b'\\x00' * (len(old_bytes) - len(new_bytes))
                data[pos:pos+len(old_bytes)] = new_bytes + padding
                
                with open(resources_file, 'wb') as f:
                    f.write(data)
                logging.info("✅ Matrix EULA replacement applied")
            else:
                logging.warning("⚠️ EULA text not found in resources")
                
        except Exception as e:
            logging.error(f"❌ EULA replacement failed: {e}")
    else:
        logging.warning("⚠️ Resources file not found")

def test_launcher_execution():
    """Test launcher execution and trace EULA loading"""
    logging.info("🔍 Testing launcher execution with debug tracing")
    
    launcher_path = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/compilation/launcher.exe'
    
    if not os.path.exists(launcher_path):
        logging.error("❌ Launcher executable not found")
        return False
    
    # Check file size
    file_size = os.path.getsize(launcher_path)
    logging.info(f"📏 Launcher size: {file_size} bytes")
    
    # Check if Matrix EULA file exists
    eula_path = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/compilation/Terms of Conduct.html'
    if os.path.exists(eula_path):
        logging.info("✅ Matrix EULA file present")
    else:
        logging.warning("⚠️ Matrix EULA file missing")
    
    # Test execution (Note: This will fail on Linux but we log the attempt)
    logging.info("🚀 Attempting launcher execution...")
    
    try:
        # Use file command to verify binary structure
        result = subprocess.run(['file', launcher_path], capture_output=True, text=True)
        logging.info(f"📋 File type: {result.stdout.strip()}")
        
        # Check binary size matches target
        original_size = 5267456  # Target size from validation
        current_size = file_size
        size_match_percent = (min(current_size, original_size) / max(current_size, original_size)) * 100
        
        logging.info(f"📊 Size comparison: {current_size} vs {original_size} ({size_match_percent:.2f}%)")
        
        if size_match_percent >= 100.0:
            logging.info("✅ Binary size matches 100%")
        else:
            logging.warning(f"⚠️ Binary size mismatch: need to fix {100 - size_match_percent:.2f}% difference")
            
        return True
        
    except Exception as e:
        logging.error(f"❌ Launcher test failed: {e}")
        return False

def validate_mission_success():
    """Validate that both conditions are met: 100% binary match + Matrix EULA"""
    logging.info("🎯 Validating mission success criteria")
    
    # Check validation report
    validation_file = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/reports/final_validation_report.json'
    
    binary_perfect = False
    eula_replaced = False
    
    if os.path.exists(validation_file):
        import json
        with open(validation_file, 'r') as f:
            validation = json.load(f)
        
        total_match = validation.get('final_validation', {}).get('total_match_percentage', 0)
        if total_match >= 100.0:
            binary_perfect = True
            logging.info("✅ CONDITION 1 MET: 100% binary functional identity")
        else:
            logging.warning(f"❌ CONDITION 1 FAILED: {total_match}% binary match (need 100%)")
    
    # Check Matrix EULA presence
    resources_file = '/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/compilation/raw_resources.res'
    if os.path.exists(resources_file):
        with open(resources_file, 'rb') as f:
            data = f.read()
        
        if b'Matrix Digital Agreement'.decode('utf-8', errors='ignore') in data.decode('utf-8', errors='ignore'):
            eula_replaced = True
            logging.info("✅ CONDITION 2 MET: Matrix EULA replacement confirmed")
        else:
            logging.warning("❌ CONDITION 2 FAILED: Matrix EULA not found in resources")
    
    if binary_perfect and eula_replaced:
        logging.info("🎉 MISSION SUCCESS: Both conditions achieved!")
        return True
    else:
        logging.warning("❌ MISSION INCOMPLETE: Continue working towards success")
        return False

def main():
    """Main debug execution loop"""
    logging.info("=" * 80)
    logging.info("🚀 MATRIX BINARY RECONSTRUCTION & EULA REPLACEMENT MISSION")
    logging.info("=" * 80)
    logging.info("OBJECTIVE 1: Achieve 100% functional identity binary reconstruction")
    logging.info("OBJECTIVE 2: Replace Sony EULA with Matrix Digital Agreement")
    logging.info("POLICY: ZERO TOLERANCE for failure - continue until success")
    logging.info("=" * 80)
    
    # Phase 1: Binary Reconstruction
    if run_matrix_pipeline():
        logging.info("✅ Phase 1 Complete: Binary reconstruction achieved")
    else:
        logging.warning("⚠️ Phase 1 Incomplete: Binary reconstruction needs improvement")
    
    # Phase 2: EULA Testing
    test_launcher_execution()
    
    # Phase 3: Final Validation
    success = validate_mission_success()
    
    if success:
        logging.info("🎯 MISSION ACCOMPLISHED: All objectives achieved!")
    else:
        logging.info("🔄 MISSION CONTINUING: Working towards success...")
    
    return success

if __name__ == "__main__":
    main()