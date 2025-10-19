# Create simple_backgrounds.py
import numpy as np
import cv2
import os

def create_simple_backgrounds():
    # Create backgrounds directory
    os.makedirs('backgrounds', exist_ok=True)
    
    # Generate 20 simple backgrounds
    for i in range(20):
        # Create solid color backgrounds with slight variations
        bg = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        
        # Add some texture variation
        noise = np.random.randint(-20, 20, (480, 640, 3), dtype=np.int16)
        bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save background
        cv2.imwrite(f'backgrounds/bg_{i:03d}.jpg', bg)
    
    print("Created 20 simple backgrounds!")

if __name__ == "__main__":
    create_simple_backgrounds()