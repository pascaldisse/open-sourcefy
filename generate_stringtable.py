#!/usr/bin/env python3
"""
Generate Windows RC STRINGTABLE from extracted string files.
This script processes all string_*.txt files and creates a comprehensive
STRINGTABLE section for inclusion in a Windows .rc resource script.
"""

import os
import re
import glob
from pathlib import Path

def escape_rc_string(text):
    """
    Escape special characters for RC file format.
    """
    # Remove line number prefix if present (format: spaces + number + tab)
    if '\t' in text and text.strip().startswith(('1→', '2→', '3→', '4→', '5→', '6→', '7→', '8→', '9→')):
        # Find the tab after the line number and take everything after it
        parts = text.split('\t', 1)
        if len(parts) > 1:
            text = parts[1]
    
    # Remove arrow prefix if still present
    if '→' in text:
        text = text.split('→', 1)[-1]
    
    # Clean up the text
    text = text.strip()
    
    # Replace backslashes with double backslashes
    text = text.replace('\\', '\\\\')
    
    # Replace double quotes with escaped quotes
    text = text.replace('"', '\\"')
    
    # Replace newlines and other control characters
    text = text.replace('\n', '\\n')
    text = text.replace('\r', '\\r')
    text = text.replace('\t', '\\t')
    
    # Handle other control characters by converting to hex escapes
    result = ""
    for char in text:
        if ord(char) < 32 or ord(char) > 126:
            # Convert non-printable characters to hex escape
            result += f"\\x{ord(char):02X}"
        else:
            result += char
    
    return result

def extract_file_number(filename):
    """
    Extract the numeric part from string_XXXX.txt filename.
    """
    match = re.match(r'string_(\d+)\.txt', filename)
    if match:
        return int(match.group(1))
    return 0

def main():
    # Path to the string files directory
    string_dir = "/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/output/launcher/20250615-130632/agents/agent_07_keymaker/resources/string"
    
    # Output file path
    output_file = "/mnt/c/Users/pascaldisse/Downloads/open-sourcefy/stringtable_resources.rc"
    
    if not os.path.exists(string_dir):
        print(f"Error: Directory {string_dir} does not exist")
        return 1
    
    # Get all string files and sort them numerically
    string_files = []
    for filename in os.listdir(string_dir):
        if filename.startswith('string_') and filename.endswith('.txt'):
            string_files.append(filename)
    
    # Sort by numeric value
    string_files.sort(key=extract_file_number)
    
    print(f"Found {len(string_files)} string files")
    
    # Generate the STRINGTABLE
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("// Generated STRINGTABLE for Windows RC file\n")
        f.write("// Source: Agent 07 Keymaker extracted strings\n")
        f.write(f"// Total entries: {len(string_files)}\n")
        f.write("// Resource IDs start from 3001\n\n")
        
        f.write("STRINGTABLE\nBEGIN\n")
        
        resource_id = 3001
        processed_count = 0
        
        for filename in string_files:
            filepath = os.path.join(string_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as string_file:
                    content = string_file.read()
                    
                    # Skip empty files
                    if not content.strip():
                        continue
                    
                    # Escape the content for RC format
                    escaped_content = escape_rc_string(content)
                    
                    # Skip if the escaped content is empty or too short
                    if len(escaped_content.strip()) == 0:
                        continue
                    
                    # Write the resource entry
                    f.write(f'    {resource_id}    "{escaped_content}"\n')
                    
                    resource_id += 1
                    processed_count += 1
                    
                    # Progress indicator
                    if processed_count % 1000 == 0:
                        print(f"Processed {processed_count} files...")
                        
            except Exception as e:
                print(f"Warning: Could not process {filename}: {e}")
                continue
        
        f.write("END\n\n")
        f.write(f"// Total processed entries: {processed_count}\n")
        f.write(f"// Resource ID range: 3001 - {resource_id - 1}\n")
    
    print(f"Generated STRINGTABLE with {processed_count} entries")
    print(f"Resource IDs: 3001 - {resource_id - 1}")
    print(f"Output saved to: {output_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())