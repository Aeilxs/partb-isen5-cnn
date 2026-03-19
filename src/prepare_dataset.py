import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

def prepare_enrico(raw_dir, processed_dir, target_classes, split_ratio=0.7):
    # 1. Loading labels
    csv_path = os.path.join(raw_dir, "design_topics.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # The Enrico CSV uses 'topic' and 'screen_id'
    df = pd.read_csv(csv_path)

    # 2. Filtering classes (we only keep the 6 that interest us)
    # Note: Check if it's lowercase or uppercase in the CSV (often lowercase)
    df['topic'] = df['topic'].str.capitalize()
    df = df[df['topic'].isin(target_classes)]

    print(f"Total number of images after filtering: {len(df)}")

    # 3. Split Train/Test (70/30)
    train_df, test_df = train_test_split(
        df,
        train_size=split_ratio,
        stratify=df['topic'],
        random_state=42
    )

    # 4. Copy files into the right folders
    for set_name, current_df in [("train", train_df), ("test", test_df)]:
        print(f"Creating set : {set_name}...")
        for _, row in current_df.iterrows():
            target_dir = os.path.join(processed_dir, set_name, row['topic'])
            os.makedirs(target_dir, exist_ok=True)

            # In Enrico, the file is just the ID.jpg
            src_path = os.path.join(raw_dir, "screenshots", f"{row['screen_id']}.jpg")
            dst_path = os.path.join(target_dir, f"{row['screen_id']}.jpg")

            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Missing img : {src_path}")

if __name__ == "__main__":
    CLASSES = ["List", "Tutorial", "Gallery", "Login"]
    prepare_enrico(
        raw_dir="data/raw/enrico",
        processed_dir="data/processed",
        target_classes=CLASSES
    )
    print("Done! Data is ready in data/processed/")