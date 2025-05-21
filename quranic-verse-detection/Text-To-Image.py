import os
import pandas as pd
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

# === Load dataset ===
excel_path = r"Dataset-Verse-by-Verse.xlsx"
df = pd.read_excel(excel_path)

# === Create output folder ===
output_folder = "image"
os.makedirs(output_folder, exist_ok=True)

# === Process each Ayah ===
for index, row in df.iterrows():
    surah_no = row['SurahNo']
    ayah_no = row['AyahNo']
    arabic_text = str(row['OrignalArabicText'])

    # Reshape Arabic text for correct display
    reshaped_text = arabic_reshaper.reshape(arabic_text)
    bidi_text = get_display(reshaped_text)

    # Create figure
    plt.figure(figsize=(10, 2))
    plt.text(0.5, 0.5, bidi_text, fontsize=24, ha='center', va='center', fontname='Arial', color='black')
    plt.axis('off')

    # Save the image
    filename = f"{surah_no:03d}_{ayah_no:03d}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.3)
    plt.close()

    print(f"Saved: {filepath}")

print("✅ All images saved to 'image/' folder.")
