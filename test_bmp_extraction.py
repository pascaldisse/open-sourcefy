#!/usr/bin/env python3
"""
Direct test of Agent 8 BMP extraction capabilities
"""

import sys
sys.path.append('src')

from src.core.agents.agent08_keymaker_resource_reconstruction import Agent8_Keymaker_ResourceReconstruction
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_bmp_extraction():
    """Test BMP extraction directly"""
    print("🖼️ Testing Matrix Online BMP extraction...")
    
    # Create Agent 8 instance
    agent8 = Agent8_Keymaker_ResourceReconstruction()
    
    # Test the BMP extraction method directly
    binary_path = "/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/input/launcher.exe"
    
    if not Path(binary_path).exists():
        print(f"❌ Binary not found: {binary_path}")
        return
    
    print(f"📁 Analyzing binary: {binary_path}")
    print(f"📏 Binary size: {Path(binary_path).stat().st_size:,} bytes")
    
    # Test image extraction
    print("\n🔍 Starting BMP extraction...")
    bmp_resources = agent8._extract_pe_image_resources(binary_path)
    
    print(f"\n📊 Extraction Results:")
    print(f"   Total BMP images found: {len(bmp_resources)}")
    
    # Save extracted BMPs to test directory
    output_dir = Path("output/test_bmp_extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, resource in enumerate(bmp_resources):
        if resource.resource_type in ['image', 'icon']:  # Include both image and icon types
            resource_file = output_dir / resource.name
            with open(resource_file, 'wb') as f:
                f.write(resource.content)
            
            print(f"   ✅ {resource.resource_type.upper()} {i+1}: {resource.metadata.get('width', '?')}x{resource.metadata.get('height', '?')} pixels")
            print(f"      📄 File: {resource.name}")
            print(f"      📏 Size: {resource.size:,} bytes") 
            print(f"      📍 Offset: 0x{resource.metadata.get('offset', 0):08X}")
            print()
    
    print(f"💾 BMPs saved to: {output_dir.absolute()}")
    
    # Also test the string extraction we know works
    print("\n🔤 Testing string extraction for comparison...")
    string_resources = agent8._extract_pe_string_table(binary_path)
    print(f"   Total strings found: {len(string_resources)}")
    
    # Look for Matrix Online strings
    matrix_strings = [s for s in string_resources if 'matrix' in str(s.content).lower()]
    print(f"   Matrix-related strings: {len(matrix_strings)}")
    
    for ms in matrix_strings[:5]:  # First 5
        print(f"      🎯 {ms.content}")

if __name__ == "__main__":
    test_bmp_extraction()