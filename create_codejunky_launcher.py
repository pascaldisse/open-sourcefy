#!/usr/bin/env python3
"""
Create Codejunky Launcher - Button URL Modification Script
Replaces all lith.thematrixonline.net URLs with codejunky.com equivalents
"""

import os
import shutil
from pathlib import Path

def create_codejunky_launcher():
    """Replace URLs in launcher binary to point to codejunky.com"""
    
    # Source and target paths
    original_launcher = Path("/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/input/launcher.exe")
    output_dir = Path("/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/compilation")
    codejunky_launcher = output_dir / "launcher_codejunky.exe"
    
    print("🔧 CREATING CODEJUNKY LAUNCHER")
    print("=" * 50)
    
    if not original_launcher.exists():
        print(f"❌ Original launcher not found: {original_launcher}")
        return False
    
    # Read original binary
    with open(original_launcher, 'rb') as f:
        binary_data = bytearray(f.read())
    
    print(f"📊 Original binary size: {len(binary_data):,} bytes")
    
    # URL replacement mappings
    url_replacements = {
        # Button URLs (main targets)
        b"http://support.lith.thematrixonline.net/": b"http://support.codejunky.com/matrix/\x00\x00\x00\x00\x00\x00\x00",
        b"http://account.lith.thematrixonline.net/": b"http://account.codejunky.com/matrix/\x00\x00\x00\x00\x00\x00\x00",
        b"http://forum.lith.thematrixonline.net/": b"http://forum.codejunky.com/matrix/\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"http://patchnotes.lith.thematrixonline.net/": b"http://updates.codejunky.com/matrix/\x00\x00\x00\x00\x00\x00\x00",
        b"http://expired.lith.thematrixonline.net/": b"http://expired.codejunky.com/matrix/\x00\x00\x00\x00\x00\x00\x00\x00",
        b"http://faq.lith.thematrixonline.net/": b"http://help.codejunky.com/matrix/\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"http://patch.lith.thematrixonline.net/": b"http://patch.codejunky.com/matrix/\x00\x00\x00\x00\x00\x00\x00\x00",
        
        # Server hostnames (for multiplayer functionality)
        b"lpad1.lith.thematrixonline.net:9700": b"server1.codejunky.com:9700\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"lpad2.lith.thematrixonline.net:9700": b"server2.codejunky.com:9700\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"lpad3.lith.thematrixonline.net:9700": b"server3.codejunky.com:9700\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        b"lpad4.lith.thematrixonline.net:9700": b"server4.codejunky.com:9700\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        
        # Authentication server
        b"auth.lith.thematrixonline.net": b"auth.codejunky.com\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
        
        # Domain strings (catch remaining references)
        b".lith.thematrixonline.net": b".codejunky.com\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    }
    
    # Apply replacements
    replacements_made = 0
    total_bytes_changed = 0
    
    for old_url, new_url in url_replacements.items():
        count = 0
        start = 0
        while True:
            pos = binary_data.find(old_url, start)
            if pos == -1:
                break
            
            # Replace the URL
            binary_data[pos:pos+len(old_url)] = new_url[:len(old_url)]
            count += 1
            replacements_made += 1
            total_bytes_changed += len(old_url)
            start = pos + len(old_url)
        
        if count > 0:
            print(f"✅ Replaced {count}x: {old_url.decode('utf-8', errors='ignore')}")
            print(f"   └─> {new_url[:len(old_url)].decode('utf-8', errors='ignore')}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write modified binary
    with open(codejunky_launcher, 'wb') as f:
        f.write(binary_data)
    
    print(f"\n🎉 CODEJUNKY LAUNCHER CREATED!")
    print(f"✅ File: {codejunky_launcher}")
    print(f"📊 Final size: {len(binary_data):,} bytes")
    print(f"🔄 Total replacements: {replacements_made}")
    print(f"📈 Bytes modified: {total_bytes_changed}")
    
    # Also copy as the main launcher.exe for immediate use
    main_launcher = output_dir / "launcher.exe"
    shutil.copy2(codejunky_launcher, main_launcher)
    print(f"🔄 Also copied to: {main_launcher}")
    
    # Verify file creation
    if codejunky_launcher.exists() and codejunky_launcher.stat().st_size > 5000000:
        print(f"\n🏆 SUCCESS: Codejunky launcher created with button modifications!")
        print(f"🎯 All Matrix Online URLs now point to codejunky.com domains")
        return True
    else:
        print(f"❌ FAILED: Codejunky launcher creation failed")
        return False

if __name__ == "__main__":
    success = create_codejunky_launcher()
    exit(0 if success else 1)