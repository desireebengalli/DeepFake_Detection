import pandas as pd
import matplotlib.pyplot as plt
import os

csv_path = "/home/giadapoloni/results2/CLIP5_values_per_epoch/clip_auc_per_epoch.csv"

save_dir = "/home/giadapoloni/visualizations/epoch_graph"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "video_auc_per_epoch.png")

df = pd.read_csv(csv_path)

plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["video_auc_roc"], marker='o', label="Video AUC-ROC")

plt.xlabel("Epoch")
plt.ylabel("AUC-ROC")
plt.title("Video AUC-ROC for Epoch")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(save_path)
plt.close()

print(f"Graph Saved in: {save_path}")
