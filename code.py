import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from pathlib import Path
import os
from typing import Dict
import tqdm

def compute_features(h) -> Dict[str, float]:
    mag, ang = [], []
    for l in range(len(h)-1):
        delta = h[l+1] - h[l]
        layer_mag = np.linalg.norm(delta.reshape(-1))
        mag.append(layer_mag)
        
        curr_flat = h[l].reshape(-1, h[l].shape[-1])
        next_flat = h[l+1].reshape(-1, h[l+1].shape[-1])
        
        angles = []
        for i in range(curr_flat.shape[0]):
            v1 = curr_flat[i] / np.linalg.norm(curr_flat[i])
            v2 = next_flat[i] / np.linalg.norm(next_flat[i])
            cos_sim = np.dot(v1, v2)
            angles.append(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        
        ang.append(np.mean(angles))
    
    delta_z = h[-1] - h[0]
    z_mag = np.linalg.norm(delta_z.reshape(-1))
    
    first_flat = h[0].reshape(-1, h[0].shape[-1])
    last_flat = h[-1].reshape(-1, h[-1].shape[-1])
    
    z_angles = []
    for i in range(first_flat.shape[0]):
        v1 = first_flat[i] / np.linalg.norm(first_flat[i])
        v2 = last_flat[i] / np.linalg.norm(last_flat[i])
        cos_sim = np.dot(v1, v2)
        z_angles.append(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
    
    z_ang = np.mean(z_angles)
    
    normalized_mag = np.mean(mag) / z_mag * 100 if z_mag != 0 else 0
    normalized_ang = np.mean(ang) / z_ang * 100 if z_ang != 0 else 0
    
    return {
        'mag': normalized_mag,
        'ang': normalized_ang
    }

def read_data_from_csv(csv_file, method_name):
    try:
        df = pd.read_csv(csv_file)
        data_dict = {}
        
        expected_columns = ['Method', 'Protein', 'pLDDT', 'GDT', 'TMscore', 'RMSD', 'LDDT']
        if not all(col in df.columns for col in expected_columns):
            print(f"Warning: CSV does not contain all expected columns: {expected_columns}")
            print(f"Found columns: {df.columns.tolist()}")
        
        method_rows = df[df['Method'] == method_name]
        
        for _, row in method_rows.iterrows():
            protein = row['Protein'].lower()  
            if pd.notna(row['pLDDT']) and pd.notna(row['RMSD']):
                lddt_value = float(row['pLDDT']) if row['pLDDT'] != 0 else float(row['pLDDT']) / 100.0
                rmsd_value = float(row['RMSD'])
                
                data_dict[protein] = {
                    'lddt': lddt_value,
                    'rmsd': rmsd_value
                }
            
        print(f"Read data for {len(data_dict)} proteins from CSV")
        return data_dict
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}

def analyze_pt_files(pt_dir, data_dict, output_dir, method_name, lddt_coef_range=range(-10, 11), mag_coef_range=range(-10, 11)):
    """Analyze PT files and correlate with RMSD values using different coefficient combinations for lddt and mag"""
    
    pt_dir_path = Path(pt_dir)
    all_pt_files = list(pt_dir_path.glob("**/*_ccd.pt"))
    print(f"Found {len(all_pt_files)} PT files")
    
    # Collect basic data first
    base_stats = {
        'mag': [],
        'ang': [],
        'lddt': [],
        'rmsd': [],
        'names': []
    }
    
    for pt_file in all_pt_files:
        try:
            file_name = pt_file.stem
            file_name = file_name.lower().split('_')[0]
            
            if file_name not in data_dict:
                print(f"Warning: No data for file {file_name}")
                continue
            
            data_e = torch.load(pt_file, map_location=torch.device('cpu'))
            data_e = [tensor.cpu() for tensor in data_e]
            num_layers = len(data_e)
            batch_size, seq_length, dim = data_e[0].shape
            
            combined_array = np.zeros((num_layers, batch_size, seq_length, dim))
            for i, layer_output in enumerate(data_e):
                combined_array[i] = layer_output.numpy()
            
            features = compute_features(combined_array)
            
            mag_value = float(features['mag'])
            ang_value = float(features['ang'])
            
            lddt_value = data_dict[file_name]['lddt'] / 100
            rmsd_value = data_dict[file_name]['rmsd']
            
            base_stats['mag'].append(mag_value)
            base_stats['ang'].append(ang_value)
            base_stats['lddt'].append(lddt_value)
            base_stats['rmsd'].append(rmsd_value)
            base_stats['names'].append(file_name)
                
        except Exception as e:
            print(f"Error processing file {pt_file}: {e}")
    
    num_processed = len(base_stats['names'])
    print(f"\nSuccessfully processed {num_processed} files")
    
    if num_processed == 0:
        print("No files were successfully processed. Analysis terminated.")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track the best coefficient combination based on absolute Pearson correlation
    best_pearson_r = 0
    best_lddt_coef = 0
    best_mag_coef = 0
    best_combined_scores = []
    best_pearson_p = 0
    best_spearman_r = 0
    best_spearman_p = 0
    
    # Loop through all combinations of lddt and mag coefficients
    for lddt_coef in lddt_coef_range:
        for mag_coef in mag_coef_range:
            # Skip (0,0) case to avoid division by zero
            if lddt_coef == 0 and mag_coef == 0:
                continue
                
            # Calculate combined score based on the coefficients
            combined_scores = []
            for i in range(num_processed):
                combined_score = lddt_coef * base_stats['lddt'][i] + mag_coef * base_stats['mag'][i]
                combined_scores.append(combined_score)
            
            # Calculate correlation with RMSD
            pearson_r, pearson_p = pearsonr(combined_scores, base_stats['rmsd'])
            spearman_r, spearman_p = spearmanr(combined_scores, base_stats['rmsd'])
            
            # Track the best coefficient combination based on absolute Pearson correlation
            if abs(spearman_r) > abs(best_spearman_r):
                best_pearson_r = pearson_r
                best_pearson_p = pearson_p
                best_spearman_r = spearman_r
                best_spearman_p = spearman_p
                best_lddt_coef = lddt_coef
                best_mag_coef = mag_coef
                best_combined_scores = combined_scores.copy()
    
    # Create a DataFrame with only the best results
    best_result_df = pd.DataFrame({
        'target': base_stats['names'],
        'mag': base_stats['mag'],
        'ang': base_stats['ang'],
        'lddt': base_stats['lddt'],
        'rmsd': base_stats['rmsd'],
        'best_combined_score': best_combined_scores
    })
    
    # Save only the best result to CSV
    best_result_df.to_csv(f'{output_dir}/{method_name}_best_result.csv', index=False)
    
    # Save best coefficient information to a separate file
    best_coef_df = pd.DataFrame({
        'lddt_coef': [best_lddt_coef],
        'mag_coef': [best_mag_coef],
        'pearson_r': [best_pearson_r],
        'pearson_p': [best_pearson_p],
        'spearman_r': [best_spearman_r],
        'spearman_p': [best_spearman_p]
    })
    best_coef_df.to_csv(f'{output_dir}/{method_name}_best_coefficient.csv', index=False)
    
    print(f"Best result saved to '{output_dir}/{method_name}_best_result.csv'")
    print(f"Best coefficient information saved to '{output_dir}/{method_name}_best_coefficient.csv'")
    
    # Print best coefficient combination
    print("\nBest coefficient combination based on absolute Pearson correlation:")
    print(f"LDDT coefficient: {best_lddt_coef}")
    print(f"MAG coefficient: {best_mag_coef}")
    print(f"Combined score formula: {best_lddt_coef} * LDDT + {best_mag_coef} * MAG")
    print(f"Pearson correlation: r={best_pearson_r:.4f}, p={best_pearson_p:.4f}")
    print(f"Spearman correlation: r={best_spearman_r:.4f}, p={best_spearman_p:.4f}")
    
    return best_result_df, best_coef_df

if __name__ == "__main__":
    # Replace with actual paths
    csv_file = "/data/home/luruiqiang/guchunbin/coe/mgd_results/updated.csv"
    pt_dir = '/data/home/luruiqiang/guchunbin/coe/mgd_boltz1_pred'
    output_dir = './mgd_new_rmsd'
    method_name = 'mgd_results'  
    
    # Define coefficient ranges for lddt and mag range(-10, 11)
    lddt_coef_range = range(-10,11) # From -10 to 10 inclusive
    mag_coef_range = range(-10,11)  # From -10 to 10 inclusive
    
    # Read data from CSV
    data_dict = read_data_from_csv(csv_file, method_name)
    
    # Analyze PT files and correlate with RMSD using different coefficient combinations
    if data_dict:
        results = analyze_pt_files(pt_dir, data_dict, output_dir, method_name, lddt_coef_range, mag_coef_range)
    else:
        print("Failed to get data from CSV file. Analysis terminated")