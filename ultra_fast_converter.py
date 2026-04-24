import os
import struct
import numpy as np
import matplotlib.pyplot as plt
# --- STEP 0: CUDA ENVIRONMENT FIX (BRUTE FORCE) ---
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
if os.path.exists(cuda_path):
    os.environ["CUDA_PATH"] = cuda_path
    bin_path = os.path.join(cuda_path, "bin")
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(bin_path)

import cupy as cp
def analyze_distribution(db_path, sample_size=200000):
    if not os.path.exists(db_path):
        return None, "Database file not found."
    
    phis = []
    rs = []
    
    with open(db_path, 'rb') as f:
        file_size = os.path.getsize(db_path)
        while len(phis) < sample_size:
            header = f.read(4)
            if not header:
                break
            num_windows = struct.unpack('<I', header)[0]
            
            phi_bytes = f.read(num_windows * 8 * 2)
            r_bytes = f.read(num_windows * 8 * 2)
            
            if len(phi_bytes) < num_windows * 8 * 2:
                break
                
            phi_data = np.frombuffer(phi_bytes, dtype=np.float16)
            r_data = np.frombuffer(r_bytes, dtype=np.float16)
            
            # Subsample within the read to avoid biased local correlations
            indices = np.random.choice(len(phi_data), min(len(phi_data), 1000), replace=False)
            phis.extend(phi_data[indices])
            rs.extend(r_data[indices])
            
            if f.tell() >= file_size:
                break

    return np.array(phis), np.array(rs)

db_file = 'master.polardb'
phis, rs = analyze_distribution(db_file)
import os, struct, numpy as np
import pandas as pd

def find_db_overlaps(db_path, rounding_precision=2):
    """
    Identifies high-frequency collisions in Angle/Magnitude space.
    Rounding to 2 decimal places simulates the resolution of 
    a high-quality 6-bit quantization.
    """
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return None

    data_list = []
    print(f"Scanning {db_path} for overlaps...")

    with open(db_path, 'rb') as f:
        # Scan up to 1 million windows for a representative sample
        while len(data_list) < 1000000:
            header = f.read(4)
            if not header: break
            n_win = struct.unpack('<I', header)[0]
            
            # Read phi and r
            phi = np.frombuffer(f.read(n_win * 16), dtype=np.float16).astype(np.float32)
            r = np.frombuffer(f.read(n_win * 16), dtype=np.float16).astype(np.float32)
            
            # Pack into list for DataFrame processing
            # Rounding is key here to identify "clusters"
            for p, m in zip(phi, r):
                data_list.append([round(p, rounding_precision), round(m, rounding_precision)])

    df = pd.DataFrame(data_list, columns=['angle', 'magnitude'])
    
    # Calculate Overlaps
    overlap_df = df.groupby(['angle', 'magnitude']).size().reset_index(name='occurrences')
    overlap_df = overlap_df.sort_values(by='occurrences', ascending=False).reset_index(drop=True)
    
    return overlap_df

# --- EXECUTION ---
overlaps = find_db_overlaps('master.polardb')

if overlaps is not None:
    print("\n--- TOP 20 OVERLAPPING SIGNAL POINTS ---")
    print(overlaps.head(100).to_string(index=False))
    # Assuming 'overlaps' is your DataFrame from the previous run
    signal_only = overlaps[overlaps['magnitude'] > 0.4].sort_values(by='occurrences', ascending=False)

    print("\n--- TOP 20 TRUE GENOMIC SIGNALS (Magnitude > 0.4) ---")
    print(signal_only.head(20).to_string(index=False))
    # Optional: Save to CSV for external analysis
    overlaps.to_csv('polardb_overlaps.csv', index=False)
if phis is not None:
    # 1. Histogram of Angles
    plt.figure(figsize=(10, 5))
    plt.hist(phis, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Angles ($\phi$) in master.polardb')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('angle_histogram.png')
    plt.close()

    # 2. Polar Histogram
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, projection='polar')
    # binning angles
    counts, bin_edges = np.histogram(phis, bins=64, range=(-np.pi, np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    width = np.diff(bin_edges)
    
    bars = ax.bar(bin_centers, counts, width=width, bottom=0.0, color='salmon', edgecolor='black', alpha=0.6)
    ax.set_title('Polar Distribution of Angles', va='bottom')
    plt.savefig('angle_polar_dist.png')
    plt.close()
    
    # 3. Angle vs Magnitude (Heatmap/Scatter)
    plt.figure(figsize=(10, 6))
    plt.hexbin(phis, rs, gridsize=50, cmap='YlOrRd')
    plt.colorbar(label='Count')
    plt.title('Angular Density vs Magnitude (Confidence)')
    plt.xlabel('Angle ($\phi$)')
    plt.ylabel('Magnitude ($r$)')
    plt.savefig('angle_magnitude_hexbin.png')
    plt.close()

    print(f"Analysis complete. Sampled {len(phis)} points.")
else:
    print("Failed to analyze distribution.")

import os, struct, numpy as np
import cupy as cp
from tqdm import tqdm

# --- 1. DEFINE YOUR CUSTOM 18S CENTROIDS ---
# These are taken directly from your Top 20 Signal-Centric Overlaps
# Centroid 0 is reserved for 'Null/Noise'
SIGNAL_CENTROIDS = np.array([
    0.0,    # Index 0: Reserved for noise
    1.67, 2.12, -1.47, 1.57, -2.00, -1.52, -1.65, 1.85, 
    1.47, -1.44, -1.72, 2.06, 2.21, 1.31, -1.57
], dtype=np.float32)

def run_quantization_v3(input_db, output_db):
    print(f"Applying PolarQuant (arXiv:2502.02617) logic to {input_db}...")
    centroids_gpu = cp.array(SIGNAL_CENTROIDS)
    
    with open(input_db, 'rb') as fin, open(output_db, 'wb') as fout:
        # Write the codebook to the header so the search engine knows the bins
        fout.write(struct.pack('<16f', *SIGNAL_CENTROIDS))
        
        while True:
            header = fin.read(4)
            if not header: break
            n_win = struct.unpack('<I', header)[0]
            
            # Read original float16 data
            phi = np.frombuffer(fin.read(n_win * 16), dtype=np.float16).astype(np.float32)
            r = np.frombuffer(fin.read(n_win * 16), dtype=np.float16).astype(np.float32)
            
            # GPU-Accelerated Centroid Mapping
            phi_gpu = cp.array(phi)
            # Find closest centroid index
            distances = cp.abs(phi_gpu[:, None] - centroids_gpu)
            phi_idx = cp.argmin(distances, axis=1).astype(cp.uint8).get()
            
            # 4-bit Magnitude Mapping (0-15)
            r_idx = np.clip(r * 15, 0, 15).astype(np.uint8)
            
            # Special Rule: If magnitude is very low, force to Null Bin (0)
            phi_idx[r < 0.15] = 0
            
            # Pack into single byte [Phi(4b) | R(4b)]
            packed = (phi_idx << 4) | (r_idx & 0x0F)
            
            fout.write(struct.pack('<I', n_win))
            fout.write(packed.tobytes())

    print(f"Quantization Complete: {os.path.getsize(output_db)/1024**2:.1f} MB")

# --- 2. THE LUT SEARCH ENGINE ---
class QuantizedSearchEngine:
    def __init__(self, q_db_path):
        with open(q_db_path, 'rb') as f:
            self.centroids = cp.array(struct.unpack('<16f', f.read(64)))
            print("Loading Quantized Blocks into VRAM...")
            all_data = []
            while True:
                h = f.read(4)
                if not h: break
                n = struct.unpack('<I', h)[0]
                all_data.append(np.frombuffer(f.read(n * 8), dtype=np.uint8).reshape(-1, 8))
        self.vram_data = cp.array(np.vstack(all_data))

    def search(self, motif, threshold=0.75):
        # Calculate query phi signature using previous logic
        q_phi = get_phi_sig(motif) 
        
        # Pre-calculate Cosine LUT (Look-Up Table)
        # This is the "PolarQuant" optimization: math is done once per search
        cos_lut = cp.cos(self.centroids - q_phi)
        
        # Extract packed 4-bit values
        v_phi_idx = self.data_vram >> 4
        v_r_val = (self.data_vram & 0x0F) / 15.0
        
        # Instant Table Lookup search
        scores = cp.mean(cp.take(cos_lut, v_phi_idx) * v_r_val, axis=1)
        return cp.where(scores > threshold)[0]

if __name__ == "__main__":
    run_quantization_v3('master.polardb', 'master_quantized.polardb')