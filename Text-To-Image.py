import os
import pandas as pd
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display
from matplotlib import font_manager
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

# === Label list ===
label_data = []

# === Process each Ayah ===
for index, row in df.iterrows():
    surah_no = row['SurahNo']
    ayah_no = row['AyahNo']
    juz = row['Juz']
    surah_name = row['SurahNameEnglish']
    classification = row['Classification']  # Makki/Madani
    arabic_text = str(row['OrignalArabicText'])

    # Reshape Arabic text
    reshaped_text = arabic_reshaper.reshape(arabic_text)
    bidi_text = get_display(reshaped_text)

    # Create figure
    plt.figure(figsize=(10, 3))
    # Add Arabic font (download and specify path)
    arabic_font = r"D:\Group Project\PY CODE\quranic-verse-detection\Amiri-Regular.ttf" # Explicit extension
    plt.text(0.5, 0.6, bidi_text, fontsize=24, ha='center', va='center', fontproperties=font_manager.FontProperties(fname=arabic_font))
    plt.text(0.5, 0.15, f"Surah: {surah_name} | Ayah: {ayah_no} | Juz: {juz} | {classification}", 
             fontsize=12, ha='center', va='center', color='gray')
    plt.axis('off')

    # Save image
    filename = f"{surah_no:03d}_{ayah_no:03d}.png"
    filepath = os.path.join(output_folder, filename)
    plt.savefig(filepath, bbox_inches='tight', pad_inches=0.3)
    plt.close()

    # Save label info
    label_data.append({"filename": filename, "surah_name": surah_name})

# === Save label CSV ===
pd.DataFrame(label_data).to_csv("ayah_labels.csv", index=False)
print("✅ All images and labels saved.")
