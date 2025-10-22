import os
import os.path as osp
from glob import glob

import uproot as uproot
import awkward as ak
import numpy as np
import cv2

from tqdm import tqdm

import torch
import matplotlib.pyplot as plt


def load_branch_with_highest_cycle(file, branch_name):
    # use this to load the tree if some of file.keys() are duplicates ending with different numbers

    # Get all keys in the file
    all_keys = file.keys()

    # Filter keys that match the specified branch name
    matching_keys = [
        key for key in all_keys if key.startswith(branch_name)]

    if not matching_keys:
        raise ValueError(
            f"No branch with name '{branch_name}' found in the file.")

    # Find the key with the highest cycle
    highest_cycle_key = max(matching_keys, key=lambda key: int(key.split(";")[1]))

    # Load the branch with the highest cycle
    branch = file[highest_cycle_key]

    return branch


if __name__ == "__main__":
    base_folder = "/home/czeh"
    hist_folder = osp.join(base_folder, "hits")
    data_folder = osp.join(base_folder, "hitsData")
    os.makedirs(data_folder, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'avc1', etc.

    low_res_folder = osp.join(data_folder, "low_res")
    high_res_folder = osp.join(data_folder, "high_res")
    video_folder = osp.join(data_folder, "videos")
    os.makedirs(low_res_folder, exist_ok=True)
    os.makedirs(high_res_folder, exist_ok=True)
    os.makedirs(video_folder, exist_ok=True)

    files = glob(f"{hist_folder}/*.root")

    pixel_opts = {"low_res": 0.2, "high_res": 0.05}

    file = uproot.open(files[0])
    metadata_lines = []

    rechits = load_branch_with_highest_cycle(file, 'ticlDumper/rechits').arrays()
    simhits = load_branch_with_highest_cycle(file, 'ticlDumper/simhits').arrays()
    clusters = load_branch_with_highest_cycle(file, 'ticlDumper/clusters').arrays()
    tracksters = load_branch_with_highest_cycle(file, 'ticlDumper/ticlTrackstersCLUE3DHigh').arrays()
    # Workaround until we have radius
    # rechits["radius"] = ak.full_like(rechits["ID"], 0.5)
    radius = 0.5

    rechits_field_map = {name: i for i, name in enumerate(rechits.fields)}
    print(rechits.fields)
    print(simhits.fields)
    print(clusters.fields)
    for i in range(len(rechits)):
        # event_folder = osp.join(data_folder, f"event_{i}")
        # os.makedirs(event_folder, exist_ok=True)

        event_rechits = rechits[i]
        event_rechits = ak.Array([event_rechits[field] for field in rechits.fields])

        event_clusters = clusters[i]
        event_tracksters = tracksters[i]

        # Convert directly to Torch tensor
        event_rechits = ak.to_torch(event_rechits).T
        event_rechits = event_rechits[torch.argsort(event_rechits[:, rechits_field_map["position_z"]])] 
        sigma = torch.Tensor([0.7 * radius]) 
        max_sigma = sigma.max().item()
        pad = max(3.0 * max_sigma, pixel_opts["low_res"])  # 3 sigma padding to capture most gaussians
        grid_pad = max(3.0 * max_sigma, 8) * pixel_opts["low_res"]  # 3 sigma padding to capture most gaussians

        xmin = event_rechits[:, rechits_field_map["position_x"]].min() - pad
        xmax = event_rechits[:, rechits_field_map["position_x"]].max() + pad
        ymin = event_rechits[:, rechits_field_map["position_y"]].min() - pad
        ymax = event_rechits[:, rechits_field_map["position_y"]].max() + pad
        
        for trackster_id in range(event_tracksters.NTracksters):
            all_layer_hits = []
            min_r = 0
            min_phi = 0

            if (event_tracksters["barycenter_z"][trackster_id] < 0):
                layers = list(range(24))
            else:
                layers = list(range(24, 48))

            for layer in layers:
                total_energy_points = 0
                layer_hits = []

                for cluster in event_tracksters["vertices_indexes"][trackster_id]:
                    if (event_clusters["cluster_layer_id"][cluster] != layer):
                        continue
                    rechits_grid = event_rechits[torch.isin(event_rechits[:, rechits_field_map["ID"]], torch.tensor(ak.values_astype(event_clusters["rechits"][cluster], np.int64)))]
                    layer_hits.append(rechits_grid)
                    total_energy_points += rechits_grid[:, rechits_field_map["energy"]].sum()

                    for hit in rechits_grid:
                        dx = hit[rechits_field_map["position_x"]] 
                        dy = hit[rechits_field_map["position_y"]]
                        r2 = dx * dx + dy * dy
                        
                        if (min_r > r2):
                            min_r = r2
                            min_phi = np.arctan2(hit[rechits_field_map["position_y"]], hit[rechits_field_map["position_x"]])

                if (len(layer_hits) > 0):
                    all_layer_hits.append(torch.cat(layer_hits))
                else:
                    # add correct shape to allow for cat
                    all_layer_hits.append(torch.empty((0, event_rechits.shape[1])))

            all_hits = torch.cat(all_layer_hits) 
            if (all_hits.shape[0] == 0):
                continue


            x = all_hits[:, rechits_field_map["position_x"]]
            y = all_hits[:, rechits_field_map["position_y"]]
            all_hits[:, rechits_field_map["position_x"]] = x * np.cos(min_phi) - y * np.sin(min_phi)
            all_hits[:, rechits_field_map["position_y"]] = x * np.sin(min_phi) + y * np.cos(min_phi)

            min_xy = torch.min(all_hits[:, rechits_field_map["position_x"]:rechits_field_map["position_y"]+1], axis=0).values - grid_pad
            max_xy = torch.max(all_hits[:, rechits_field_map["position_x"]:rechits_field_map["position_y"]+1], axis=0).values + grid_pad


            for name, pixel in pixel_opts.items():
                trackster_folder_name = f"event_{i}_trackster_{trackster_id}"
                trackster_folder = osp.join(data_folder, name, trackster_folder_name)
                os.makedirs(trackster_folder, exist_ok=True)
                # pixel_folder = osp.join(trackster_folder, name)
                # os.makedirs(pixel_folder, exist_ok=True)

                # diff_x = 12.8/pixel - (max_xy[0] - min_xy[0])
                # print(12.8/pixel)
                # print((max_xy[0] - min_xy[0]))
                # if (diff_x > 0):
                #     min_xy[0] -= diff_x/2
                #     max_xy[0] += diff_x/2
                #
                # print((max_xy[0] - min_xy[0]))
                # diff_y = 12.8/pixel - (max_xy[1] - min_xy[1])
                # if (diff_y > 0):
                #     min_xy[1] -= diff_y/2
                #     max_xy[1] += diff_y/2

                # Use pixel-aligned edges
                x_edges = torch.arange(torch.floor(xmin / pixel) * pixel, torch.ceil(xmax / pixel) * pixel + pixel, pixel)
                y_edges = torch.arange(torch.floor(ymin / pixel) * pixel, torch.ceil(ymax / pixel) * pixel + pixel, pixel)
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

                x_mask = (x_centers >= min_xy[0]) & (x_centers <= max_xy[0])
                y_mask = (y_centers >= min_xy[1]) & (y_centers <= max_xy[1])

                # Grid centers mesh
                Xc, Yc = torch.meshgrid(x_centers, y_centers, indexing="xy")

                # ---- Gaussian splatting ----
                # Energy-normalized 2D isotropic Gaussian: density = E / (2*pi*sigma^2) * exp(-r^2/(2*sigma^2))
                # Pixel energy is approximated as density(center) * pixel_area
                pix_area = pixel * pixel

                if (Xc[y_mask][:, x_mask].shape[0] == 0 or Xc[:, x_mask].shape[1] == 0):
                    continue

                # maybe edd missing on both sides, not just one
                if (name == "high_res"):
                    print(w*4, Xc[y_mask][:, x_mask].shape[1])
                    if (h*4 != Xc[y_mask][:, x_mask].shape[0]):
                        diff = Xc[y_mask][:, x_mask].shape[0] - h*4 
                        y_mask = (y_centers >= (min_xy[1]+diff*pixel)) & (y_centers <= max_xy[1])
                    if (w*4 != Xc[y_mask][:, x_mask].shape[1]):
                        diff = Xc[y_mask][:, x_mask].shape[1] - w*4 
                        x_mask = (x_centers >= (min_xy[0]+diff*pixel)) & (x_centers <= max_xy[0])
                    print(w*4, Xc[y_mask][:, x_mask].shape[1])

                grids = []
                for layer, hits in zip(layers, all_layer_hits):
                    grid = torch.zeros_like(Xc)
                    total_energy_points = hits[:, rechits_field_map["energy"]].sum()
                    x = hits[:, rechits_field_map["position_x"]]
                    y = hits[:, rechits_field_map["position_y"]]
                    hits[:, rechits_field_map["position_x"]] = x * np.cos(min_phi) - y * np.sin(min_phi)
                    hits[:, rechits_field_map["position_y"]] = x * np.sin(min_phi) + y * np.cos(min_phi)

                    for hit in hits:
                        dx = Xc - hit[rechits_field_map["position_x"]]
                        dy = Yc - hit[rechits_field_map["position_y"]]
                        r2 = dx * dx + dy * dy
                        # si = 0.7 * hit[rechits_field_map["radius"]]
                        si = 0.7 * 0.5
                        density = (hit[rechits_field_map["energy"]] / (2.0 * np.pi * si * si)) * torch.exp(-r2 / (2.0 * si * si))
                        grid += density * pix_area

                    # Optional: compute total energy in grid for sanity check
                    total_energy_grid = grid.sum()
                    # print(layer, total_energy_grid, total_energy_points)

                    grid_cut = grid[y_mask][:, x_mask].cpu().numpy()
                    grid_norm = cv2.normalize(grid_cut, None, 0, 255, cv2.NORM_MINMAX)
                    #comment line for training!
                    # grid_norm = cv2.bitwise_not(grid_norm)

                    grid_uint8 = grid_norm.astype(np.uint8)

                    pad_h = max(0, int(12.8/pixel) - grid_uint8.shape[0])
                    pad_w = max(0, int(12.8/pixel) - grid_uint8.shape[1])

                    # Pad bottom and right
                    grid_uint8 = cv2.copyMakeBorder(
                        grid_uint8,
                        top=pad_h, bottom=0,
                        left=0, right=pad_w,
                        borderType=cv2.BORDER_CONSTANT,
                        value=0  # fill with black
                    )

                    grids.append(grid_uint8)
                    cv2.imwrite(osp.join(trackster_folder, f"{layer:05d}.png"), grid_uint8)


                mv_path = osp.join(video_folder, f"event_{i}_trackster_{trackster_id}_{name}.mp4")
                fps = 5  # frames per second
                h, w = grids[0].shape[:2]
                print(w, h)

                out = cv2.VideoWriter(mv_path, fourcc, fps, (w, h), isColor=False)
                for frame in grids:
                    out.write(frame)

                out.release()
            metadata_lines.append(f"{trackster_folder_name} {len(layers)} ({grid_uint8.shape[0]},{grid_uint8.shape[1]},1) {layers[0]:05d}")

    metadata_path = osp.join(data_folder, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata_lines))

