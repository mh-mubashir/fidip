# Create simple_backgrounds.py
import numpy as np
import cv2
import os

def create_simple_backgrounds():
    # Create backgrounds directory
    os.makedirs('backgrounds', exist_ok=True)
    
    # Define clean, simple background colors
    background_colors = [
        (240, 240, 240),  # Light gray
        (220, 220, 220),  # Medium gray
        (200, 200, 200),  # Darker gray
        (255, 250, 240),  # Cream
        (245, 245, 245),  # Off-white
        (230, 230, 230),  # Light gray
        (210, 210, 210),  # Medium gray
        (190, 190, 190),  # Darker gray
        (255, 255, 250),  # Very light cream
        (235, 235, 235),  # Light gray
        (225, 225, 225),  # Medium gray
        (215, 215, 215),  # Darker gray
        (250, 250, 250),  # Almost white
        (245, 245, 240),  # Cream
        (230, 230, 225),  # Light gray
        (220, 220, 215),  # Medium gray
        (210, 210, 205),  # Darker gray
        (255, 255, 255),  # Pure white
        (240, 240, 235),  # Light cream
        (235, 235, 230),  # Medium cream
    ]
    
    # Generate 20 clean backgrounds
    for i in range(20):
        # Create solid color background
        bg = np.full((480, 640, 3), background_colors[i], dtype=np.uint8)
        
        # Add very subtle gradient for depth (optional)
        if i % 3 == 0:  # Every 3rd background gets a subtle gradient
            for y in range(480):
                factor = 0.95 + 0.1 * (y / 480)  # Subtle vertical gradient
                bg[y, :] = np.clip(bg[y, :] * factor, 0, 255).astype(np.uint8)
        
        # Save background
        cv2.imwrite(f'backgrounds/bg_{i:03d}.jpg', bg)
    
    print("Created 20 clean, simple backgrounds!")

if __name__ == "__main__":
    create_simple_backgrounds()