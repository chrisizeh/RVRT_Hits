import os
import re
import os.path as osp
from glob import glob
import math

import uproot
import awkward as ak
import numpy as np
import cv2

from tqdm import tqdm

import torch
import torch.nn.functional as F


def load_branch_with_highest_cycle(file, branch_name):
    # Find all keys that match the branch name with any cycle
    all_keys = file.keys()
    matching_keys = [key for key in all_keys if key.startswith(branch_name)]
    if not matching_keys:
        raise ValueError(f"No branch with name '{branch_name}' found in the file.")
    # Find the key with the highest cycle
    highest_cycle_key = max(matching_keys, key=lambda key: int(key.split(";")[1]))
    # Load the branch with the highest cycle
    branch = file[highest_cycle_key]
    return branch


def choose_targets_from_lowres(H, W, multiple=8, min_size=64):
    Ht = max(min_size, math.ceil(H / multiple) * multiple)
    Wt = max(min_size, math.ceil(W / multiple) * multiple)
    return Ht, Wt


def crop_pad_center_2d(grid_hw, target_h, target_w, pad_value=0.0):
    """grid_hw: (H, W) tensor. Return (1, target_h, target_w) with center-crop and symmetric pad."""
    H, W = grid_hw.shape
    # Center crop if larger
    if H > target_h:
        top = (H - target_h) // 2
        grid_hw = grid_hw[top:top + target_h, :]
        H = target_h
    if W > target_w:
        left = (W - target_w) // 2
        grid_hw = grid_hw[:, left:left + target_w]
        W = target_w
    # Symmetric pad if smaller
    pad_h = max(0, target_h - H)
    pad_w = max(0, target_w - W)
    pad_top    = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left   = pad_w // 2
    pad_right  = pad_w - pad_left
    x = grid_hw.unsqueeze(0)  # (1, H, W)
    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=pad_value)
    return x  # (1, target_h, target_w)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    base_folder = "/home/czeh"

    hist_folder = osp.join(base_folder, "hits")
    train_data_folder = osp.join(base_folder, "hitsDataTrain")
    test_data_folder = osp.join(base_folder, "hitsData")
    os.makedirs(train_data_folder, exist_ok=True)
    os.makedirs(test_data_folder, exist_ok=True)

    max_count = 5500
    test_count = 500
    data_folder = test_data_folder

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')

    low_res_folder = osp.join(data_folder, "low_res")
    high_res_folder = osp.join(data_folder, "high_res")
    video_folder = osp.join(data_folder, "videos")
    os.makedirs(low_res_folder, exist_ok=True)
    os.makedirs(high_res_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    files = glob(f"{hist_folder}/histoRechits.root")

    pixel_opts = {"low_res": 0.2, "high_res": 0.05}

    file = uproot.open(files[0])
    metadata_lines = []

    rechits = load_branch_with_highest_cycle(file, 'ticlDumper/rechits').arrays()
    clusters = load_branch_with_highest_cycle(file, 'ticlDumper/clusters').arrays()
    tracksters = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTrackstersCLUE3DHigh').arrays()

    # Workaround until we have radius
    # rechits["radius"] = ak.full_like(rechits["ID"], 0.5)
    radius = 0.5

    rechits_field_map = {name: i for i, name in enumerate(rechits.fields)}
    print(rechits.fields)
    print(clusters.fields)
    for i in range(min(max_count, len(rechits))):
        if (i == test_count):
            data_folder = train_data_folder

        pattern = re.compile(f"^event_{i}_.+$")

        if any(pattern.match(d) for d in os.listdir(low_res_folder) if os.path.isdir(os.path.join(low_res_folder, d))):
            print(f"Skip event {i}")
            continue
        else:
            print(f"Starting event {i}")

        event_rechits = rechits[i]
        event_rechits = ak.Array([event_rechits[field] for field in rechits.fields])

        event_clusters = clusters[i]
        event_tracksters = tracksters[i]

        # Convert directly to Torch tensor
        event_rechits = ak.to_torch(event_rechits).T.to(device)
        event_rechits = event_rechits[torch.argsort(event_rechits[:, rechits_field_map["position_z"]])]
        sigma = torch.tensor([0.7 * radius], device=device)
        max_sigma = sigma.max().item()
        pad = max(3.0 * max_sigma, pixel_opts["low_res"])  # 3 sigma padding to capture most gaussians
        grid_pad = max(3.0 * max_sigma, 8) * pixel_opts["low_res"]  # 3 sigma padding to capture most gaussians

        xmin = event_rechits[:, rechits_field_map["position_x"]].min() - pad
        xmax = event_rechits[:, rechits_field_map["position_x"]].max() + pad
        ymin = event_rechits[:, rechits_field_map["position_y"]].min() - pad
        ymax = event_rechits[:, rechits_field_map["position_y"]].max() + pad  # fixed typo

        for trackster_id in range(event_tracksters.NTracksters):
            all_layer_hits = []
            all_sim_layer_hits = []
            min_r = 0
            min_phi = 0

            if (event_tracksters["barycenter_z"][trackster_id] < 0):
                layers = list(range(24))
            else:
                layers = list(range(24, 48))

            for layer in layers:
                total_energy_points = 0
                total_sim_energy_points = 0
                layer_hits = []
                sim_layer_hits = []

                for cluster in event_tracksters["vertices_indexes"][trackster_id]:
                    if (event_clusters["cluster_layer_id"][cluster] != layer):
                        continue
                    rechits_grid = event_rechits[torch.isin(
                        event_rechits[:, rechits_field_map["ID"]],
                        torch.tensor(ak.values_astype(event_clusters["rechits"][cluster], np.int64), device=device)
                    )]
                    simhits_grid = rechits_grid.clone()
                    simhits_grid[:, rechits_field_map["energy"]] = simhits_grid[:, rechits_field_map["simEnergy"]]

                    layer_hits.append(rechits_grid)
                    sim_layer_hits.append(simhits_grid)
                    total_energy_points += rechits_grid[:, rechits_field_map["energy"]].sum()
                    total_sim_energy_points += simhits_grid[:, rechits_field_map["energy"]].sum()

                    cluster_sim_hits = []
                    for hit in rechits_grid:

                        dx = hit[rechits_field_map["position_x"]]
                        dy = hit[rechits_field_map["position_y"]]
                        r2 = dx * dx + dy * dy

                        if (min_r > r2):
                            min_r = r2
                            min_phi = np.arctan2(hit[rechits_field_map["position_y"]], hit[rechits_field_map["position_x"]])

                # print(f"rec energy: {total_energy_points}, sim energy: {total_sim_energy_points}")
                if (len(layer_hits) > 0):
                    all_layer_hits.append(torch.cat(layer_hits))
                    all_sim_layer_hits.append(torch.cat(sim_layer_hits))
                else:
                    # add correct shape to allow for cat
                    all_layer_hits.append(torch.empty((0, event_rechits.shape[1]), device=device))
                    all_sim_layer_hits.append(torch.empty((0, event_rechits.shape[1]), device=device))


            all_hits = torch.cat(all_layer_hits)
            if (all_hits.shape[0] == 0):
                continue

            x = all_hits[:, rechits_field_map["position_x"]]
            y = all_hits[:, rechits_field_map["position_y"]]
            all_hits[:, rechits_field_map["position_x"]] = x * np.cos(min_phi) - y * np.sin(min_phi)
            all_hits[:, rechits_field_map["position_y"]] = x * np.sin(min_phi) + y * np.cos(min_phi)

            min_xy = torch.min(all_hits[:, rechits_field_map["position_x"]:rechits_field_map["position_y"]+1], axis=0).values - grid_pad
            max_xy = torch.max(all_hits[:, rechits_field_map["position_x"]:rechits_field_map["position_y"]+1], axis=0).values + grid_pad

            # ---- Force order: low_res first, then high_res ----
            passes = [('low_res', pixel_opts['low_res'], all_layer_hits), ('high_res', pixel_opts['high_res'], all_sim_layer_hits)]
            lowres_targets = {}

            for name, pixel, specific_layer_hits in passes:
                trackster_folder_name = f"event_{i}_trackster_{trackster_id}"
                trackster_folder = osp.join(data_folder, name, trackster_folder_name)
                os.makedirs(trackster_folder, exist_ok=True)

                # Use pixel-aligned edges
                x_edges = torch.arange(torch.floor(xmin / pixel) * pixel, torch.ceil(xmax / pixel) * pixel + pixel, pixel, device=device)
                y_edges = torch.arange(torch.floor(ymin / pixel) * pixel, torch.ceil(ymax / pixel) * pixel + pixel, pixel, device=device)
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

                x_mask = (x_centers >= min_xy[0]) & (x_centers <= max_xy[0])
                y_mask = (y_centers >= min_xy[1]) & (y_centers <= max_xy[1])

                # Grid centers mesh
                Xc, Yc = torch.meshgrid(x_centers, y_centers, indexing="xy")

                # Skip if empty
                if (Xc[y_mask][:, x_mask].shape[0] == 0 or Xc[:, x_mask].shape[1] == 0):
                    continue

                # Decide targets
                if name == 'low_res':
                    H_low = int(Xc[y_mask][:, x_mask].shape[0])
                    W_low = int(Xc[y_mask][:, x_mask].shape[1])
                    target_h_low, target_w_low = choose_targets_from_lowres(H_low, W_low, multiple=8, min_size=64)
                    lowres_targets[trackster_folder_name] = (target_h_low, target_w_low)
                    target_h, target_w = target_h_low, target_w_low
                else:
                    if trackster_folder_name not in lowres_targets:
                        # Fallback (shouldn't happen)
                        H_hi = int(Xc[y_mask][:, x_mask].shape[0])
                        W_hi = int(Xc[y_mask][:, x_mask].shape[1])
                        target_h_low, target_w_low = choose_targets_from_lowres(math.ceil(H_hi/4), math.ceil(W_hi/4), multiple=8, min_size=64)
                    else:
                        target_h_low, target_w_low = lowres_targets[trackster_folder_name]
                    target_h, target_w = 4 * target_h_low, 4 * target_w_low

                grids = []
                for layer, hits in zip(layers, specific_layer_hits):
                    grid = torch.zeros_like(Xc, device=device)
                    pix_area = pixel * pixel

                    if hits.numel() > 0:
                        x = hits[:, rechits_field_map["position_x"]]
                        y = hits[:, rechits_field_map["position_y"]]
                        hits[:, rechits_field_map["position_x"]] = x * np.cos(min_phi) - y * np.sin(min_phi)
                        hits[:, rechits_field_map["position_y"]] = x * np.sin(min_phi) + y * np.cos(min_phi)

                        for hit in hits:
                            dx = Xc - hit[rechits_field_map["position_x"]]
                            dy = Yc - hit[rechits_field_map["position_y"]]
                            r2 = dx * dx + dy * dy
                            si = 0.7 * 0.5
                            density = (hit[rechits_field_map["energy"]] / (2.0 * np.pi * si * si)) * torch.exp(-r2 / (2.0 * si * si))
                            grid += density * pix_area

                    grid_cut = grid[y_mask][:, x_mask].cpu()  # (H, W)
                    grid_cut = crop_pad_center_2d(grid_cut, target_h, target_w, pad_value=0.0)  # (1, target_h, target_w)

                    grids.append(grid_cut)
                    torch.save(grid_cut, osp.join(trackster_folder, f"{layer:05d}.pt"))

                # Metadata (use H,W from shaped grid)
                if grids:
                    H_meta, W_meta = grids[0].shape[1], grids[0].shape[2]
                    metadata_lines.append(f"{trackster_folder_name} {len(layers)} ({H_meta},{W_meta},1) {layers[0]:05d}")

    metadata_path = osp.join(data_folder, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata_lines))

