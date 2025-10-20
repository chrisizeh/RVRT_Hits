import os
import os.path as osp
from glob import glob

import uproot as uproot
import awkward as ak
import numpy as np

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

    pixel_opts = {"low_res": 0.2, "high_res": 0.05}

    file = uproot.open(files[0])

    rechits = load_branch_with_highest_cycle(file, 'ticlDumper/rechits').arrays()
    simhits = load_branch_with_highest_cycle(file, 'ticlDumper/simhits').arrays()
    clusters = load_branch_with_highest_cycle(file, 'ticlDumper/clusters').arrays()
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
        print("event_clusters", event_clusters)

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

        # ---- Define grid bounds ----

        # Use pixel-aligned edges
        for name, pixel in pixel_opts.items():
            pixel_folder = osp.join(event_folder, name)
            os.makedirs(pixel_folder, exist_ok=True)

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

            for layer in layers:
                # grid_cluster = event_clusters[event_clusters["cluster_layer_id"] == layer]
                grid = torch.zeros_like(Xc)
                total_energy_points = 0

                for cluster in range(len(event_clusters["energy"])):
                    if (event_clusters["cluster_layer_id"][cluster] != layer):
                        continue

                    # print(event_rechits[:, rechits_field_map["ID"]])
                    # print(torch.tensor(ak.values_astype(event_clusters["rechits"][cluster], np.int32)))
                    # print(torch.isin(event_rechits[:, rechits_field_map["ID"]], torch.tensor(ak.values_astype(event_clusters["rechits"][cluster], np.int32))))
                    rechits_grid = event_rechits[torch.isin(event_rechits[:, rechits_field_map["ID"]], torch.tensor(ak.values_astype(event_clusters["rechits"][cluster], np.int32)))]
                    total_energy_points += rechits_grid[:, rechits_field_map["energy"]].sum()


                    # print("grid", grid)
                    for hit in rechits_grid:
                        # print(hit)
                        # compute squared distance to this point
                        dx = Xc - hit[rechits_field_map["position_x"]] 
                        dy = Yc - hit[rechits_field_map["position_y"]]
                        r2 = dx * dx + dy * dy
                        # si = 0.7 * hit[rechits_field_map["radius"]]
                        si = 0.7 * 0.5
                        density = (hit[rechits_field_map["energy"]] / (2.0 * np.pi * si * si)) * torch.exp(-r2 / (2.0 * si * si))
                        # print(dx, dy, r2, si)
                        # print("density", hit[rechits_field_map["energy"]], (2.0 * np.pi * si * si), torch.exp(-r2 / (2.0 * si * si)))
                        grid += density * pix_area
                        # print("grid", grid)

                # Optional: compute total energy in grid for sanity check
                total_energy_grid = grid.sum()
                print(total_energy_grid, total_energy_points)

                fig2, ax2 = plt.subplots(figsize=(6, 6))
                ax2.set_aspect("equal")
                ax2.set_title("After: rasterized energy on 0.1 grid")
                im = ax2.imshow(
                    grid,
                    origin="lower",
                    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                    interpolation="nearest",
                )
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                cbar = fig2.colorbar(im, ax=ax2)
                cbar.set_label("Energy per pixel")

                plt.savefig(osp.join(pixel_folder, f"grid_{layer}"))
                plt.close()


