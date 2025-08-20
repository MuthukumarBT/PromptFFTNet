import os
import re
import csv
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

gt_root = '/home/uib42091/Muthu/heatChamberTest/original'#"/home/uib42091/Muthu/heatChamberTest/original/"
out_root = "/home/uib42091/Muthu/PromptIR/Demo3HeatChamber1"
csv_path = "psnr_ssim_results_Demo2Heatchamber1.csv"  # output CSV file

# Helper to extract numeric ID
def get_id(filename):
    return int(re.findall(r'\d+', filename)[0])

# Store overall results
all_psnr_vals = []
all_ssim_vals = []

# Store folder-wise results
folder_results = {}

# Iterate through subfolders
for subfolder in sorted(os.listdir(gt_root)):
    gt_dir = os.path.join(gt_root, subfolder)
    out_dir = os.path.join(out_root, subfolder)

    if not os.path.isdir(gt_dir) or not os.path.isdir(out_dir):
        continue

    # Build dictionaries indexed by ID
    gt_images = {get_id(f): f for f in os.listdir(gt_dir) if f.endswith('.jpg')}
    out_images = {get_id(f): f for f in os.listdir(out_dir) if f.endswith('.png')}

    # Common image IDs
    common_ids = sorted(set(gt_images.keys()) & set(out_images.keys()))

    psnr_vals = []
    ssim_vals = []

    for img_id in tqdm(common_ids, desc=f"Evaluating {subfolder}", leave=False):
        gt_path = os.path.join(gt_dir, gt_images[img_id])
        out_path = os.path.join(out_dir, out_images[img_id])

        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        out_img = cv2.imread(out_path, cv2.IMREAD_COLOR)

        if gt_img is None or out_img is None:
            print(f"⚠️ Error reading image ID {img_id} in folder {subfolder}")
            continue

        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)

        if gt_img.shape != out_img.shape:
            out_img = cv2.resize(out_img, (gt_img.shape[1], gt_img.shape[0]))

        psnr_val = psnr(gt_img, out_img, data_range=255)
        ssim_val = ssim(gt_img, out_img, channel_axis=-1, data_range=255)

        psnr_vals.append(psnr_val)
        ssim_vals.append(ssim_val)

    if psnr_vals:
        avg_psnr = sum(psnr_vals) / len(psnr_vals)
        avg_ssim = sum(ssim_vals) / len(ssim_vals)
        folder_results[subfolder] = (avg_psnr, avg_ssim)

        all_psnr_vals.extend(psnr_vals)
        all_ssim_vals.extend(ssim_vals)

# Write CSV
with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Folder', 'Average PSNR', 'Average SSIM'])

    print("\n📁 Folder-wise PSNR and SSIM:")
    for folder, (psnr_val, ssim_val) in folder_results.items():
        print(f"  {folder}: PSNR = {psnr_val:.4f}, SSIM = {ssim_val:.4f}")
        writer.writerow([folder, f"{psnr_val:.4f}", f"{ssim_val:.4f}"])

    if all_psnr_vals:
        avg_psnr_all = sum(all_psnr_vals) / len(all_psnr_vals)
        avg_ssim_all = sum(all_ssim_vals) / len(all_ssim_vals)

        print(f"\n✅ Overall Average PSNR: {avg_psnr_all:.4f}")
        print(f"✅ Overall Average SSIM: {avg_ssim_all:.4f}")

        writer.writerow([])
        writer.writerow(['Overall Average', f"{avg_psnr_all:.4f}", f"{avg_ssim_all:.4f}"])
    else:
        print("❌ No valid image pairs found.")
