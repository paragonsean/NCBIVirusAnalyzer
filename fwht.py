import os
import numpy as np
import zipfile
from tqdm import tqdm

# --- STEP 0: CUDA ENVIRONMENT INITIALIZATION ---
# Update this path to your actual CUDA installation
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
if os.path.exists(cuda_path):
    os.environ["CUDA_PATH"] = cuda_path
    # Add bin to DLL path for Windows Store Python compatibility
    bin_path = os.path.join(cuda_path, "bin")
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(bin_path)

import cupy as cp

# --- STEP 1: POLAR TRANSFORMATION LOGIC ---
BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

def fwht(a):
    """Fast Walsh-Hadamard Transform (Recursive)"""
    n = len(a)
    if n == 1: return a
    half = n // 2
    left, right = fwht(a[0:half]), fwht(a[half:n])
    return np.concatenate([left + right, left - right])

def get_motif_signature(motif, block_size=8):
    """Converts a query string into a polar angle signature."""
    z = [1.0 * np.exp(1j * PHASES[BASES[b]]) for b in motif[:block_size]]
    # Precondition (Rotation)
    rot_real, rot_imag = fwht(np.real(z)), fwht(np.imag(z))
    return cp.array(np.arctan2(rot_imag, rot_real))

def fastq_to_polar_blocks(seq, qual_str, block_size=8):
    """Transforms FASTQ reads into magnitude and angle blocks."""
    z_list = []
    for char, q_char in zip(seq, qual_str):
        if char not in BASES: continue
        m = (ord(q_char) - 33 + 1) / 94.0 # Magnitude from Phred
        z_list.append(m * np.exp(1j * PHASES[BASES[char]]))
    
    pad = (block_size - len(z_list) % block_size) % block_size
    z_arr = np.pad(np.array(z_list), (0, pad), constant_values=0j)
    blocks = z_arr.reshape(-1, block_size)
    
    r_list, phi_list = [], []
    for block in blocks:
        rot_real, rot_imag = fwht(block.real), fwht(block.imag)
        r_list.append(np.sqrt(rot_real**2 + rot_imag**2))
        phi_list.append(np.arctan2(rot_imag, rot_real))
        
    return cp.array(r_list), cp.array(phi_list)

# --- STEP 2: VRAM SEARCH LOGIC ---
def vram_polar_search(vram_phi, vram_r, query_phi, threshold=0.75):
    """Performs Cosine Similarity search in VRAM."""
    # Similarity = cos(phi_target - phi_query) weighted by magnitude
    diff = vram_phi - query_phi
    similarity = cp.cos(diff) * vram_r
    
    # Average similarity across the block (8 bases)
    score = cp.mean(similarity, axis=1)
    match_indices = cp.where(score > threshold)[0]
    return match_indices, score[match_indices]

# --- STEP 3: FULL FASTQ PROCESSING AND VRAM UPLOAD ---
def parse_fastq_from_zip(zip_path, filename):
    """Parse FASTQ reads from a ZIP file."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open(filename) as f:
            lines = [line.decode().strip() for line in f.readlines()]
    
    reads = []
    for i in range(0, len(lines), 4):
        if i + 3 < len(lines):
            header = lines[i]
            seq = lines[i + 1]
            qual = lines[i + 3]
            reads.append((header, seq, qual))
    return reads

def process_all_reads_to_vram(zip_path, block_size=8):
    """Process all FASTQ reads from ZIP and upload to VRAM."""
    print("Loading FASTQ files from ZIP...")
    
    all_r_blocks = []
    all_phi_blocks = []
    total_bases = 0
    
    with zipfile.ZipFile(zip_path) as z:
        fnames = [f.filename for f in z.infolist() if f.filename.endswith('.fastq')]
        # Only process first 6 files
        fnames = fnames[:6]
        
        for fname in tqdm(fnames, desc="Processing FASTQ files"):
            reads = parse_fastq_from_zip(zip_path, fname)
            
            for header, seq, qual in tqdm(reads, desc=f"Transforming {fname.split('/')[-1]}", leave=False):
                if len(seq) == 0 or len(qual) == 0:
                    continue
                
                # Transform to polar blocks
                r_blocks, phi_blocks = fastq_to_polar_blocks(seq, qual, block_size)
                
                all_r_blocks.append(r_blocks)
                all_phi_blocks.append(phi_blocks)
                total_bases += len(seq)
    
    print(f"\nConcatenating all blocks to VRAM...")
    # Concatenate all blocks and upload to VRAM
    r_gpu = cp.concatenate(all_r_blocks)
    phi_gpu = cp.concatenate(all_phi_blocks)
    
    print(f"VRAM Upload Complete:")
    print(f"  Total Bases Processed: {total_bases:,}")
    print(f"  Total VRAM Blocks: {phi_gpu.shape[0]:,}")
    print(f"VRAM Memory Used: {(r_gpu.nbytes + phi_gpu.nbytes) / 1024**2:.1f} MB")
    
    return r_gpu, phi_gpu, total_bases

# --- STEP 4: EXECUTION ON FULL DATASET ---
if __name__ == "__main__":
    ZIP_FILE = "fastqfiles.zip"
    
    # Process all reads and upload to VRAM
    r_gpu, phi_gpu, total_bases = process_all_reads_to_vram(ZIP_FILE)
    
    # Define multiple motifs to search for
    test_motifs = [
        "GTCGCTCT",  # From your example read
        "ATGCGACAG", # From your example read  
        "GTGCCGTTC", # From your example read
        "CGTGACTTC", # From your example read
        "AACAAACCA", # From your example read
    ]
    
    print(f"\n{'='*60}")
    print("POLAR MOTIF SEARCH IN VRAM")
    print(f"{'='*60}")
    
    for motif in test_motifs:
        print(f"\nSearching for motif: {motif}")
        query_signature = get_motif_signature(motif)
        
        # Scan VRAM with different thresholds
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            matches, scores = vram_polar_search(phi_gpu, r_gpu, query_signature, threshold)
            
            if len(matches) > 0:
                print(f"  Threshold {threshold}: {len(matches)} matches found")
                # Show top 5 matches
                top_indices = cp.argsort(scores)[-5:][::-1]
                for idx in top_indices:
                    print(f"    Block {idx.get()}: Score {scores[idx].get():.4f}")
                break
        else:
            print(f"  No matches found above any threshold")