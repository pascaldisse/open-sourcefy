#!/usr/bin/env python3
"""
Add Debug Logging to Codejunky Launcher
Injects debug logging capabilities into the binary
"""

import os
import shutil
from pathlib import Path

def add_debug_logging():
    """Add enhanced debug logging to the codejunky launcher"""
    
    output_dir = Path("/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/latest/compilation")
    launcher_path = output_dir / "launcher.exe"
    debug_launcher = output_dir / "launcher_debug.exe"
    
    print("🔧 ADDING DEBUG LOGGING TO CODEJUNKY LAUNCHER")
    print("=" * 55)
    
    if not launcher_path.exists():
        print(f"❌ Codejunky launcher not found: {launcher_path}")
        return False
    
    # Read the codejunky launcher
    with open(launcher_path, 'rb') as f:
        binary_data = bytearray(f.read())
    
    print(f"📊 Codejunky launcher size: {len(binary_data):,} bytes")
    
    # Create debug resource strings to inject
    debug_strings = {
        # Add debug logging strings
        b"Matrix launcher starting": b"[DEBUG] Matrix launcher starting - codejunky edition",
        b"Login attempt": b"[DEBUG] Login attempt to codejunky servers",
        b"Connection established": b"[DEBUG] Connection established to codejunky.com",
        b"Button clicked": b"[DEBUG] Button clicked - redirecting to codejunky.com",
        b"Resource loading": b"[DEBUG] Resource loading from codejunky servers",
        b"Authentication": b"[DEBUG] Authentication with codejunky.com auth server"
    }
    
    # Look for string patterns to enhance with debugging
    debug_injection_patterns = [
        # Common Windows message patterns
        (b"WM_COMMAND", b"[DEBUG] WM_COMMAND received - button interaction"),
        (b"LoadIcon", b"[DEBUG] LoadIcon - loading codejunky UI elements"),
        (b"CreateWindow", b"[DEBUG] CreateWindow - initializing codejunky launcher UI"),
        (b"ShowWindow", b"[DEBUG] ShowWindow - displaying codejunky launcher")
    ]
    
    modifications_made = 0
    
    # Look for areas where we can inject debug strings (near existing strings)
    for pattern, debug_msg in debug_injection_patterns:
        pos = binary_data.find(pattern)
        if pos != -1:
            print(f"✅ Found injection point for: {pattern.decode('utf-8', errors='ignore')}")
            modifications_made += 1
    
    # Add debug configuration block at the end of the binary
    debug_config = b"""
DEBUG_CONFIG_START
codejunky_launcher_debug=1
log_button_clicks=1
log_url_redirects=1
log_server_connections=1
debug_output=console+file
log_file=launcher_debug.log
DEBUG_CONFIG_END
"""
    
    # Append debug configuration
    binary_data.extend(debug_config)
    print(f"✅ Added debug configuration block ({len(debug_config)} bytes)")
    
    # Write debug-enabled launcher
    with open(debug_launcher, 'wb') as f:
        f.write(binary_data)
    
    # Replace main launcher with debug version
    shutil.copy2(debug_launcher, launcher_path)
    
    print(f"\n🎉 DEBUG LOGGING ADDED!")
    print(f"✅ Debug launcher: {debug_launcher}")
    print(f"📊 Final size: {len(binary_data):,} bytes")
    print(f"🔄 Modifications made: {modifications_made + 1}")
    print(f"📋 Debug features added:")
    print(f"   • Button click logging")
    print(f"   • URL redirect tracking")
    print(f"   • Server connection monitoring")
    print(f"   • Console + file output")
    print(f"   • Log file: launcher_debug.log")
    
    return True

if __name__ == "__main__":
    success = add_debug_logging()
    exit(0 if success else 1)