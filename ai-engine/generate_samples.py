"""
Generates synthetic sample multi-view test portraits for Dr. Văn Trường 3D Studio.
"""
import cv2
import numpy as np
import os

def generate_sample_portraits(output_dir="/home/ubuntu/dr-vantruong-3d-studio/samples"):
    os.makedirs(output_dir, exist_ok=True)
    
    W, H = 512, 640
    # Create sample portrait sets for Demo Patient:
    # 1. Front (Chính diện)
    # 2. Left 45 (Nghiêng trái)
    # 3. Right 45 (Nghiêng phải)
    # 4. Chin Up (Ngửa cằm)
    
    # Palette
    skin_base = np.array([170, 195, 235], dtype=np.uint8) # BGR (warm Asian skin tone)
    lip_color = np.array([120, 110, 200], dtype=np.uint8)
    hair_color = np.array([25, 25, 30], dtype=np.uint8)
    eye_dark = np.array([20, 20, 25], dtype=np.uint8)
    bg_color = np.array([245, 245, 248], dtype=np.uint8)

    slots = ["front", "left_45", "right_45", "chin_up"]
    offsets = {
        "front": (0, 0),
        "left_45": (-45, 0),
        "right_45": (45, 0),
        "chin_up": (0, -35),
    }

    for slot in slots:
        img = np.full((H, W, 3), bg_color, dtype=np.uint8)
        dx, dy = offsets[slot]
        cx, cy = W // 2 + dx, H // 2 + dy

        # Head / Face oval
        cv2.ellipse(img, (cx, cy), (140, 185), 0, 0, 360, skin_base.tolist(), -1)

        # Hair
        cv2.ellipse(img, (cx, cy - 85), (145, 120), 0, 180, 360, hair_color.tolist(), -1)
        cv2.circle(img, (cx - 110, cy - 30), 45, hair_color.tolist(), -1)
        cv2.circle(img, (cx + 110, cy - 30), 45, hair_color.tolist(), -1)

        # Eyebrows
        cv2.line(img, (cx - 80, cy - 50), (cx - 25, cy - 55), eye_dark.tolist(), 6)
        cv2.line(img, (cx + 25, cy - 55), (cx + 80, cy - 50), eye_dark.tolist(), 6)

        # Eyes
        cv2.ellipse(img, (cx - 50, cy - 30), (22, 12), 0, 0, 360, [255, 255, 255], -1)
        cv2.ellipse(img, (cx + 50, cy - 30), (22, 12), 0, 0, 360, [255, 255, 255], -1)
        cv2.circle(img, (cx - 50, cy - 30), 9, eye_dark.tolist(), -1)
        cv2.circle(img, (cx + 50, cy - 30), 9, eye_dark.tolist(), -1)
        cv2.circle(img, (cx - 52, cy - 33), 3, [255, 255, 255], -1)
        cv2.circle(img, (cx + 48, cy - 33), 3, [255, 255, 255], -1)

        # Nose
        cv2.line(img, (cx, cy - 30), (cx, cy + 30), [140, 165, 205], 4)
        cv2.ellipse(img, (cx, cy + 35), (18, 10), 0, 0, 180, [130, 150, 190], 3)

        # Cheeks blush
        overlay = img.copy()
        cv2.circle(overlay, (cx - 75, cy + 25), 35, [140, 160, 240], -1)
        cv2.circle(overlay, (cx + 75, cy + 25), 35, [140, 160, 240], -1)
        cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

        # Lips
        cv2.ellipse(img, (cx, cy + 85), (38, 16), 0, 0, 360, lip_color.tolist(), -1)
        cv2.line(img, (cx - 38, cy + 85), (cx + 38, cy + 85), [90, 80, 160], 2)

        # Save
        path = os.path.join(output_dir, f"sample_{slot}.jpg")
        cv2.imwrite(path, img)

    print(f"Generated 4 sample multi-view images in {output_dir}")

if __name__ == "__main__":
    generate_sample_portraits()

