import argparse
import torch
import numpy as np
import pandas as pd
import json
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import os
from typing import Dict


def compute_features(h):
    mag, ang = [], []
    for l in range(len(h) - 1):
        delta = h[l + 1] - h[l]
        mag.append(np.linalg.norm(delta.reshape(-1)))

        curr_flat = h[l].reshape(-1, h[l].shape[-1])
        next_flat = h[l + 1].reshape(-1, h[l + 1].shape[-1])

        angles = []
        for i in range(curr_flat.shape[0]):
            v1 = curr_flat[i] / np.linalg.norm(curr_flat[i])
            v2 = next_flat[i] / np.linalg.norm(next_flat[i])
            angles.append(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
        ang.append(np.mean(angles))

    delta_z = h[-1] - h[0]
    z_mag = np.linalg.norm(delta_z.reshape(-1))

    first_flat = h[0].reshape(-1, h[0].shape[-1])
    last_flat = h[-1].reshape(-1, h[-1].shape[-1])

    z_angles = []
    for i in range(first_flat.shape[0]):
        v1 = first_flat[i] / np.linalg.norm(first_flat[i])
        v2 = last_flat[i] / np.linalg.norm(last_flat[i])
        z_angles.append(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    z_ang = np.mean(z_angles)

    normalized_mag = np.mean(mag) / z_mag * 100 if z_mag != 0 else 0
    normalized_ang = np.mean(ang) / z_ang * 100 if z_ang != 0 else 0

    return {'mag': normalized_mag, 'ang': normalized_ang}


def load_plddt(json_dir):
    plddt_dict = {}
    for json_file in Path(json_dir).glob("**/*.json"):
        protein_id = json_file.parent.name.lower()
        with open(json_file, 'r') as f:
            data = json.load(f)
        plddt_dict[protein_id] = data['complex_plddt']
    print(f"Loaded pLDDT for {len(plddt_dict)} proteins")
    return plddt_dict


def load_rmsd(csv_path):
    df = pd.read_csv(csv_path)
    rmsd_dict = {}
    for _, row in df.iterrows():
        rmsd_dict[str(row['protein']).lower().strip()] = float(row['rmsd'])
    print(f"Loaded RMSD for {len(rmsd_dict)} proteins")
    return rmsd_dict


def analyze(pt_dir, plddt_dict, rmsd_dict, output_dir, method_name, lddt_coef_range, mag_coef_range):
    all_pt_files = list(Path(pt_dir).glob("**/*.pt"))
    print(f"Found {len(all_pt_files)} PT files")

    base_stats = {'mag': [], 'ang': [], 'lddt': [], 'rmsd': [], 'names': []}

    for pt_file in all_pt_files:
        file_name = pt_file.stem.lower().split('_')[0]

        if file_name not in plddt_dict or file_name not in rmsd_dict:
            continue

        data_e = torch.load(pt_file, map_location='cpu')
        data_e = [t.cpu() for t in data_e]
        num_layers = len(data_e)
        batch_size, seq_length, dim = data_e[0].shape

        combined_array = np.zeros((num_layers, batch_size, seq_length, dim))
        for i, layer_output in enumerate(data_e):
            combined_array[i] = layer_output.numpy()

        features = compute_features(combined_array)

        base_stats['mag'].append(float(features['mag']))
        base_stats['ang'].append(float(features['ang']))
        base_stats['lddt'].append(plddt_dict[file_name])
        base_stats['rmsd'].append(rmsd_dict[file_name])
        base_stats['names'].append(file_name)

    num_processed = len(base_stats['names'])
    print(f"Processed {num_processed} files")

    if num_processed == 0:
        return None

    os.makedirs(output_dir, exist_ok=True)

    best_spearman_r = 0
    best_pearson_r = 0
    best_pearson_p = 0
    best_spearman_p = 0
    best_lddt_coef = 0
    best_mag_coef = 0
    best_combined_scores = []

    for lddt_coef in lddt_coef_range:
        for mag_coef in mag_coef_range:
            if lddt_coef == 0 and mag_coef == 0:
                continue

            combined_scores = [
                lddt_coef * base_stats['lddt'][i] + mag_coef * base_stats['mag'][i]
                for i in range(num_processed)
            ]

            pearson_r, pearson_p = pearsonr(combined_scores, base_stats['rmsd'])
            spearman_r, spearman_p = spearmanr(combined_scores, base_stats['rmsd'])

            if abs(spearman_r) > abs(best_spearman_r):
                best_pearson_r = pearson_r
                best_pearson_p = pearson_p
                best_spearman_r = spearman_r
                best_spearman_p = spearman_p
                best_lddt_coef = lddt_coef
                best_mag_coef = mag_coef
                best_combined_scores = combined_scores

    pd.DataFrame({
        'target': base_stats['names'],
        'mag': base_stats['mag'],
        'ang': base_stats['ang'],
        'lddt': base_stats['lddt'],
        'rmsd': base_stats['rmsd'],
        'best_combined_score': best_combined_scores
    }).to_csv(f'{output_dir}/{method_name}_best_result.csv', index=False)

    pd.DataFrame({
        'lddt_coef': [best_lddt_coef],
        'mag_coef': [best_mag_coef],
        'pearson_r': [best_pearson_r],
        'pearson_p': [best_pearson_p],
        'spearman_r': [best_spearman_r],
        'spearman_p': [best_spearman_p]
    }).to_csv(f'{output_dir}/{method_name}_best_coefficient.csv', index=False)

    print(f"\nBest: lddt_coef={best_lddt_coef}, mag_coef={best_mag_coef}")
    print(f"  {best_lddt_coef} * LDDT + {best_mag_coef} * MAG")
    print(f"  Pearson  r={best_pearson_r:.4f}, p={best_pearson_p:.4f}")
    print(f"  Spearman r={best_spearman_r:.4f}, p={best_spearman_p:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pt_dir', type=str, required=True)
    parser.add_argument('--plddt_dir', type=str, required=True)
    parser.add_argument('--rmsd_csv', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--method_name', type=str, default='method')
    parser.add_argument('--lddt_coef_min', type=int, default=-10)
    parser.add_argument('--lddt_coef_max', type=int, default=10)
    parser.add_argument('--mag_coef_min', type=int, default=-10)
    parser.add_argument('--mag_coef_max', type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plddt_dict = load_plddt(args.plddt_dir)
    rmsd_dict = load_rmsd(args.rmsd_csv)
    analyze(
        args.pt_dir, plddt_dict, rmsd_dict, args.output_dir, args.method_name,
        range(args.lddt_coef_min, args.lddt_coef_max + 1),
        range(args.mag_coef_min, args.mag_coef_max + 1)
    )
