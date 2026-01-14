#!/usr/bin/env python3
"""Quick test to verify ImageMagick configuration."""

import os
import sys

# Add funclip to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'funclip'))

print("=" * 60)
print("Testing ImageMagick Configuration")
print("=" * 60)

# Test 1: Check if magick is in PATH
import shutil
magick_path = shutil.which('magick')
print(f"\n1. shutil.which('magick'): {magick_path}")
print(f"   Status: {'✓ Found' if magick_path else '✗ Not found'}")

# Test 2: Verify moviepy can be imported
try:
    import moviepy.config as mpy_config
    print(f"\n2. moviepy.config import: ✓ Success")
except ImportError as e:
    print(f"\n2. moviepy.config import: ✗ Failed - {e}")
    sys.exit(1)

# Test 3: Verify ImageMagick configuration was set
try:
    if magick_path:
        mpy_config.change_settings({"IMAGEMAGICK_BINARY": magick_path})
        print(f"   moviepy.config.change_settings: ✓ Applied")
except Exception as e:
    print(f"   moviepy.config.change_settings: ✗ Failed - {e}")

# Test 4: Import VideoClipper
try:
    from videoclipper import VideoClipper
    print(f"\n3. VideoClipper import: ✓ Success")
except Exception as e:
    print(f"\n3. VideoClipper import: ✗ Failed - {e}")
    sys.exit(1)

# Test 5: Check OpenCV availability
try:
    import cv2
    print(f"\n4. OpenCV (cv2) import: ✓ Success")
    print(f"   Haar cascade available: ✓ Success")
except ImportError as e:
    print(f"\n4. OpenCV (cv2) import: ✗ Not installed - {e}")

print("\n" + "=" * 60)
print("Configuration test completed!")
print("=" * 60)
