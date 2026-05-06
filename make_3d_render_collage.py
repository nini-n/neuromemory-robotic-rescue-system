import os
from PIL import Image, ImageDraw, ImageFont

INPUT_DIR = "outputs/advanced_3d_views"
OUTPUT_PATH = os.path.join(INPUT_DIR, "neuromemory_3d_collage.png")

image_files = [
    ("view_01_overview.png", "Overview"),
    ("view_02_top_down.png", "Top view"),
    ("view_03_side_robot_focus.png", "Robot focus"),
    ("view_04_survivor_focus.png", "Survivor focus"),
]

images = []
for fname, label in image_files:
    path = os.path.join(INPUT_DIR, fname)
    img = Image.open(path).convert("RGB")
    images.append((img, label))

target_w = 900
target_h = 560
processed = []

for img, label in images:
    img = img.resize((target_w, target_h))

    canvas = Image.new("RGB", (target_w, target_h + 70), (18, 18, 22))
    canvas.paste(img, (0, 0))

    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 32)
    except:
        font = ImageFont.load_default()

    draw.text((30, target_h + 18), label, fill=(240, 240, 240), font=font)
    processed.append(canvas)

margin = 25
title_h = 90

bg_w = target_w * 2 + margin * 3
bg_h = title_h + (target_h + 70) * 2 + margin * 3

bg = Image.new("RGB", (bg_w, bg_h), (10, 12, 16))

draw = ImageDraw.Draw(bg)

try:
    title_font = ImageFont.truetype("arial.ttf", 42)
except:
    title_font = ImageFont.load_default()

draw.text(
    (40, 25),
    "NeuroMemory Robot - Advanced 3D Rescue Scene Views",
    fill=(255, 255, 255),
    font=title_font,
)

positions = [
    (margin, title_h + margin),
    (margin * 2 + target_w, title_h + margin),
    (margin, title_h + margin * 2 + target_h + 70),
    (margin * 2 + target_w, title_h + margin * 2 + target_h + 70),
]

for img, pos in zip(processed, positions):
    bg.paste(img, pos)

bg.save(OUTPUT_PATH)
print("Saved collage:", OUTPUT_PATH)