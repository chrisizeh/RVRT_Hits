import os
from PIL import Image
import numpy as np

# Paths to your two main folders
true_path = "/home/czeh/hitsData/high_res"
reco_path = "/home/czeh/RVRT_Hits/results/002_train_rvrt_hits"

ratios = []
for subfolder in os.listdir(true_path):
    true_subpath = os.path.join(true_path, subfolder)
    reco_subpath = os.path.join(reco_path, subfolder)

    if not os.path.isdir(true_subpath) or not os.path.exists(reco_subpath):
        continue  # skip if not a matching folder

    total_true = 0
    total_reco = 0

    for filename in os.listdir(true_subpath):
        img_path1 = os.path.join(true_subpath, filename)
        img_path2 = os.path.join(reco_subpath, filename)

        if not os.path.isfile(img_path2):
            continue  # skip if not found in path2

        # Load both images
        img1 = np.array(Image.open(img_path1))
        img2 = np.array(Image.open(img_path2))

        # Add pixel sums to totals
        total_true += img1.sum()
        total_reco += img2.sum()

    # Avoid division by zero
    ratio = (total_reco / total_true) if total_true != 0 else 0
    ratios.append(ratio)

    print(f"{subfolder}: {ratio}")

mean = np.mean(ratios)  # sample standard deviation
stdev = np.std(ratios, ddof=1)  # sample standard deviation
print(f"\nMean of ratios: {mean}")
print(f"\nStandard deviation of ratios: {stdev}")


