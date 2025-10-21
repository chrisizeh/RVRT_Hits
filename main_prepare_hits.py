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

    files = glob(f"{hist_folder}/*.root")

    pixel_opts = {"low_res": 4, "high_res": 0.5}

    file = uproot.open(files[0])

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
        event_folder = osp.join(data_folder, f"event_{i}")
        os.makedirs(event_folder, exist_ok=True)

        event_rechits = rechits[i]
        event_rechits = ak.Array([event_rechits[field] for field in rechits.fields])

        event_clusters = clusters[i]
        event_tracksters = tracksters[i]

        # Convert directly to Torch tensor
        event_rechits = ak.to_torch(event_rechits).T
        event_rechits = event_rechits[torch.argsort(event_rechits[:, rechits_field_map["position_z"]])] 
        layers = list(range(48))
        sigma = torch.Tensor([0.7 * radius]) 
        max_sigma = sigma.max().item()
        pad = 3.0 * max_sigma  # 3 sigma padding to capture most gaussians
        xmin = event_rechits[:, rechits_field_map["position_x"]].min() - pad
        xmax = event_rechits[:, rechits_field_map["position_x"]].max() + pad
        ymin = event_rechits[:, rechits_field_map["position_y"]].min() - pad
        ymax = event_rechits[:, rechits_field_map["position_y"]].max() + pad
        
        for trackster_id in range(event_tracksters.NTracksters):
            trackster_folder = osp.join(event_folder, f"trackster_{trackster_id}")
            os.makedirs(trackster_folder, exist_ok=True)

            for name, pixel in pixel_opts.items():
                pixel_folder = osp.join(trackster_folder, name)
                os.makedirs(pixel_folder, exist_ok=True)

                # ---- Define grid bounds ----

                # Use pixel-aligned edges
                x_edges = torch.arange(torch.floor(xmin / pixel) * pixel, torch.ceil(xmax / pixel) * pixel + pixel, pixel)
                y_edges = torch.arange(torch.floor(ymin / pixel) * pixel, torch.ceil(ymax / pixel) * pixel + pixel, pixel)
                x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
                y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0

                # Grid centers mesh
                Xc, Yc = torch.meshgrid(x_centers, y_centers, indexing="xy")

                # ---- Gaussian splatting ----
                # Energy-normalized 2D isotropic Gaussian: density = E / (2*pi*sigma^2) * exp(-r^2/(2*sigma^2))
                # Pixel energy is approximated as density(center) * pixel_area
                pix_area = pixel * pixel
                grids = []
                all_layer_hits = []

                min_r = 0
                min_phi = 0

                for layer in layers:
                    grid = torch.zeros_like(Xc)
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
                            # add shape, but arbitrary
                            all_layer_hits.append(torch.empty((1, 1)))

                all_hits = torch.cat(all_layer_hits) 
                x = all_hits[:, rechits_field_map["position_x"]]
                y = all_hits[:, rechits_field_map["position_y"]]
                all_hits[:, rechits_field_map["position_x"]] = x * np.cos(min_phi) - y * np.sin(min_phi)
                all_hits[:, rechits_field_map["position_y"]] = x * np.sin(min_phi) + y * np.cos(min_phi)
                min_xy = torch.min(all_hits[:, rechits_field_map["position_x"]:rechits_field_map["position_y"]+1], axis=0)
                max_xy = torch.max(all_hits[:, rechits_field_map["position_x"]:rechits_field_map["position_y"]+1], axis=0)
                print(min_xy)

                for hits in all_layer_hits:
                    grid = torch.zeros_like(Xc)
                    total_energy_points = 0

                    total_energy_points = hits[:, rechits_field_map["energy"]]
                    x = hits[:, rechits_field_map["position_x"]]
                    y = hits[:, rechits_field_map["position_y"]]
                    hits[:, rechits_field_map["position_x"]] = x * np.cos(min_phi) - y * np.sin(min_phi)
                    hits[:, rechits_field_map["position_y"]] = x * np.sin(min_phi) + y * np.cos(min_phi)

                    for hit in hits:
                        dx = Xc - hit[rechits_field_map["position_x"]] - min_xy.values[0].item()
                        dy = Yc - hit[rechits_field_map["position_y"]] - min_xy.values[1].item()
                        r2 = dx * dx + dy * dy
                        # si = 0.7 * hit[rechits_field_map["radius"]]
                        si = 0.7 * 0.5
                        density = (hit[rechits_field_map["energy"]] / (2.0 * np.pi * si * si)) * torch.exp(-r2 / (2.0 * si * si))
                        grid += density * pix_area

                    # Optional: compute total energy in grid for sanity check
                    total_energy_grid = grid.sum()
                    print(total_energy_grid, total_energy_points)

                    grid_norm = cv2.normalize(grid.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX)
                    grid_uint8 = grid_norm.astype(np.uint8)
                    grids.append(grid_uint8)
                    cv2.imwrite(osp.join(pixel_folder, f"layer_{layer}.png"), grid_uint8)

                    mv_path = osp.join(pixel_folder, f"all_layers.mp4")
                    fps = 10  # frames per second
                    h, w = grids[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'avc1', etc.
                    out = cv2.VideoWriter(mv_path, fourcc, fps, (w, h), isColor=False)

                    for frame in grids:
                        if frame.ndim == 2 and not out.isOpened():
                            out = cv2.VideoWriter(mv_path, fourcc, fps, (w, h), isColor=False)
                        out.write(frame)

                    out.release()
