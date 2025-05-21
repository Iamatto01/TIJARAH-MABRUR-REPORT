import os
import pandas as pd

# === Load dataset ===
excel_path = r"Dataset-Verse-by-Verse.xlsx"
df = pd.read_excel(excel_path)

# === Create labels list ===
labels = []

# === Loop over dataset ===
for index, row in df.iterrows():
    surah_no = int(row['SurahNo'])
    ayah_no = int(row['AyahNo'])
    surah_name = row['SurahNameEnglish']

    filename = f"{surah_no:03d}_{ayah_no:03d}.png"
    labels.append({ 
        "filename": filename,
        "surah_name": surah_name,
        "ayah_no": ayah_no
    })

# === Save to CSV ===
labels_df = pd.DataFrame(labels)
labels_df.to_csv("ayah_labels.csv", index=False)

print("✅ ayah_labels.csv generated successfully.")
