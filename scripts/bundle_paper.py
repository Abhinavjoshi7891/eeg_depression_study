#!/usr/bin/env python3
import os
import shutil
import re
from pathlib import Path

# Paths
base_dir = Path("/Users/abhinavjoshi/Downloads/eeg_depression")
md_path = base_dir / "REVISED_PAPER.md"
export_dir = base_dir / "paper_export_package"
assets_dir = export_dir / "assets"

# Create directories
if export_dir.exists():
    shutil.rmtree(export_dir)
export_dir.mkdir()
assets_dir.mkdir()

# Read the markdown content
content = md_path.read_text(encoding="utf-8")

# Regex to find markdown images: ![alt_text](path/to/image.png)
img_pattern = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

def replacer(match):
    alt_text = match.group(1)
    img_path = match.group(2)
    
    # Resolve the absolute path of the image
    abs_img_path = base_dir / img_path
    
    if abs_img_path.exists():
        # Get filename
        filename = abs_img_path.name
        
        # Copy to assets_dir
        dest_path = assets_dir / filename
        shutil.copy2(abs_img_path, dest_path)
        
        # Return new markdown image tag pointing to local assets/ folder
        return f"![{alt_text}](assets/{filename})"
    else:
        print(f"Warning: Image not found: {abs_img_path}")
        return match.group(0)

# Replace all occurrences
new_content = img_pattern.sub(replacer, content)

# Write the new markdown file
out_md = export_dir / "eeg_paper_final.md"
out_md.write_text(new_content, encoding="utf-8")

# Create a ZIP package for easiest download/sharing
shutil.make_archive(str(export_dir.absolute()), 'zip', str(export_dir.absolute()))

print(f"Created export package at: {export_dir}")
print(f"Created ZIP file at: {export_dir}.zip")
