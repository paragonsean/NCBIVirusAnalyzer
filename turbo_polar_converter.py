import os
import zipfile
import struct
import numpy as np
from tqdm import tqdm

# --- STEP 0: CUDA ENVIRONMENT FIX (BRUTE FORCE) ---
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
if os.path.exists(cuda_path):
    os.environ["CUDA_PATH"] = cuda_path
    bin_path = os.path.join(cuda_path, "bin")
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(bin_path)

import cupy as cp

# --- Core Polar Logic ---
BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

def fwht(a):
    n = len(a)
    if n == 1: return a
    h = n // 2
    l, r = fwht(a[0:h]), fwht(a[h:n])
    return np.concatenate([l + r, l - r])

def process_read(seq, qual, block_size=8):
    z = []
    for b, q in zip(seq, qual):
        if b not in BASES: continue
        m = (ord(q) - 33 + 1) / 94.0
        z.append(m * np.exp(1j * PHASES[BASES[b]]))
    
    if len(z) < block_size: return None
    z = np.array(z)
    num_windows = len(z) - block_size + 1
    windows = np.lib.stride_tricks.as_strided(z, shape=(num_windows, block_size), strides=(z.strides[0], z.strides[0]))
    
    phi_list, r_list = [], []
    for w in windows:
        rr, ri = fwht(w.real), fwht(w.imag)
        r_list.append(np.sqrt(rr**2 + ri**2))
        phi_list.append(np.arctan2(ri, rr))
    
    return np.array(phi_list, dtype=np.float16), np.array(r_list, dtype=np.float16)

def convert_all_fastq(zip_path, output_db):
    total_reads = 0
    total_windows = 0
    
    with open(output_db, 'wb') as db, zipfile.ZipFile(zip_path, 'r') as z:
        fastq_files = [f for f in z.namelist() if f.endswith('.fastq')]
        
        # Outer progress bar for files
        file_pbar = tqdm(fastq_files, desc="Overall Progress", unit="file")
        
        for fname in file_pbar:
            file_pbar.set_postfix(current_file=fname[:20])
            with z.open(fname) as f:
                # We count lines to estimate read progress (4 lines per read)
                # This doesn't read the whole file into memory
                lines = []
                for line in tqdm(f, desc=f"Processing {fname[:15]}", leave=False, unit="line"):
                    lines.append(line.decode().strip())
                    if len(lines) == 4:
                        res = process_read(lines[1], lines[3])
                        if res:
                            phi, r = res
                            db.write(struct.pack('<I', len(phi)))
                            db.write(phi.tobytes())
                            db.write(r.tobytes())
                            total_reads += 1
                            total_windows += len(phi)
                        lines = []
    
    db_size = os.path.getsize(output_db)
    print(f"\nDatabase Statistics:")
    print(f"  Total Reads: {total_reads:,}")
    print(f"  Total Windows: {total_windows:,}")
    print(f"  Database Size: {db_size / 1024**2:.1f} MB")
    print(f"Finished! {total_reads} reads packed into {output_db}")

# --- Database Reader for Searching ---
class PolarDatabase:
    def __init__(self, db_path):
        self.db_path = db_path
        self.offsets = []
        self.window_counts = []
        
        db_size = os.path.getsize(db_path)
        print("Building database index...")
        
        with open(db_path, 'rb') as f:
            # Use tqdm for the indexing pass based on file size
            with tqdm(total=db_size, unit='B', unit_scale=True, desc="Indexing Database") as pbar:
                offset = 0
                while True:
                    header = f.read(4)
                    if not header: break
                    num_windows = struct.unpack('<I', header)[0]
                    self.window_counts.append(num_windows)
                    self.offsets.append(offset)
                    
                    record_size = 4 + num_windows * 2 * 2 
                    f.seek(record_size, 1)
                    offset += (4 + record_size) # Account for header + data
                    pbar.update(4 + record_size)
        
        print(f"Indexed {len(self.offsets)} reads")
    
    def search_motif(self, motif, threshold=0.7, max_results=100):
        z = [1.0 * np.exp(1j * PHASES[BASES[b]]) for b in motif]
        rr, ri = fwht(np.real(z)), fwht(np.imag(z))
        query_sig = np.arctan2(ri, rr)
        query_gpu = cp.array(query_sig, dtype=cp.float16)
        
        matches = []
        
        # tqdm for the search process
        search_pbar = tqdm(zip(self.offsets, self.window_counts), 
                          total=len(self.offsets), 
                          desc=f"Searching for '{motif}'", 
                          unit="read")
        
        with open(self.db_path, 'rb') as f:
            for read_idx, (offset, num_windows) in enumerate(search_pbar):
                f.seek(offset)
                header = f.read(4)
                if not header: break
                
                phi_bytes = f.read(num_windows * 2) 
                r_bytes = f.read(num_windows * 2)
                
                if len(phi_bytes) != num_windows * 2 or len(r_bytes) != num_windows * 2:
                    continue
                
                phi_gpu = cp.frombuffer(phi_bytes, dtype=cp.float16).reshape(-1, 8)
                r_gpu = cp.frombuffer(r_bytes, dtype=cp.float16).reshape(-1, 8)
                
                diff = phi_gpu - query_gpu
                scores = cp.mean(cp.cos(diff) * r_gpu, axis=1)
                
                match_indices = cp.where(scores > threshold)[0]
                
                if len(match_indices) > 0:
                    best_idx = int(match_indices[cp.argmax(scores[match_indices])])
                    best_score = float(scores[best_idx])
                    matches.append((read_idx, best_idx, best_score))
                    
                    if len(matches) >= max_results:
                        search_pbar.close()
                        break
        
        return matches

if __name__ == "__main__":
    convert_all_fastq('fastqfiles.zip', 'master_genome.polardb')
    
    print("\n" + "="*50)
    print("READY TO SEARCH")
    print("="*50)