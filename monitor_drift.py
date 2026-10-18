import pandas as pd
import numpy as np
import glob
from PIL import Image, ImageStat
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

def calculate_image_stats(image_path):
    try:
        img = Image.open(image_path).convert('L')
        stat = ImageStat.Stat(img)
        brightness = stat.rms[0]
        contrast = stat.stddev[0]
        img_array = np.array(img, dtype=np.int32)
        gy, gx = np.gradient(img_array)
        sharpness = np.var(gx) + np.var(gy)
        return brightness, contrast, sharpness
    except Exception:
        return None, None, None

def build_features_dataframe(image_folder, label):
    paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    paths = paths[:100] # Limit to 100 for speed
    
    data = []
    for p in paths:
        b, c, s = calculate_image_stats(p)
        if b is not None:
            data.append({"brightness": b, "contrast": c, "sharpness": s, "source": label})
            
    return pd.DataFrame(data)

def generate_evidently_report():
    print("Extracting pixel statistics from images...")
    ref_df = build_features_dataframe("data/images", "reference")
    
    # CORRECTED FOLDER NAME
    corrupted_dir = "data/corruption"
    
    if os.path.exists(corrupted_dir) and len(glob.glob(os.path.join(corrupted_dir, "*.jpg"))) > 0:
        print(f"✅ Found ACTUAL corrupted images in '{corrupted_dir}'! Calculating real drift...")
        curr_df = build_features_dataframe(corrupted_dir, "current")
    else:
        print(f"⚠️ Folder '{corrupted_dir}' is empty. Check your data.")
        return

    print("Generating Evidently Drift Report on REAL corrupted data...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df.drop(columns=['source']), 
               current_data=curr_df.drop(columns=['source']))
    
    report_path = "reports/evidently_drift_report.html"
    report.save_html(report_path)
    print(f"✅ New Evidently report saved to {report_path}")

    drift_score = np.abs(ref_df["brightness"].mean() - curr_df["brightness"].mean())
    print(f"📊 Actual Overall Pixel Brightness Drift Score: {drift_score:.2f}")
    
    if drift_score > 10.0:  # Threshold can be adjusted based on actual drift
        print("🚨 ALERT: Drift threshold exceeded! Triggering Airflow Retraining DAG...")
        with open("reports/retrain_alert.log", "w") as f:
            f.write(f"RETRAIN_TRIGGERED=TRUE\nDRIFT_SCORE={drift_score:.2f}")
    else:
        print("✅ No significant drift detected in the current data.")

if __name__ == "__main__":
    generate_evidently_report()