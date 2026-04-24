import os, zipfile, struct, numpy as np

import sys
# 1. Update this to your EXACT CUDA version folder
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
if os.path.exists(cuda_path):
    os.environ["CUDA_PATH"] = cuda_path
    # Add bin to DLL path for Windows Store Python compatibility
    bin_path = os.path.join(cuda_path, "bin")
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(bin_path)

import cupy as cp
# --- CORE POLAR LOGIC ---
BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

def fwht_iterative(a):
    """Fast Walsh-Hadamard Transform (Iterative - O(n log n))"""
    a = a.copy()
    n = len(a)
    h = 1
    while h < n:
        for i in range(0, n, 2*h):
            for j in range(i, i+h):
                a[j], a[j+h] = a[j] + a[j+h], a[j] - a[j+h]
        h *= 2
    return a

def process_read_simple(seq, qual, block_size=8):
    """Simplified polar transform without FWHT for speed"""
    z = []
    for b, q in zip(seq, qual):
        if b not in BASES: continue
        m = (ord(q) - 33 + 1) / 94.0
        z.append(m * np.exp(1j * PHASES[BASES[b]]))
    
    if len(z) < block_size: return None
    z = np.array(z)
    num_windows = len(z) - block_size + 1
    
    # Simple sliding windows without FWHT (much faster)
    phi_list, r_list = [], []
    for i in range(num_windows):
        window = z[i:i+block_size]
        # Direct polar conversion (no FWHT)
        r_list.append(np.abs(window))
        phi_list.append(np.angle(window))
    
    return np.array(phi_list, dtype=np.float16).flatten(), np.array(r_list, dtype=np.float16).flatten()

def convert_fast_subset(zip_path, output_db, max_reads=1000):
    """Convert a small subset quickly for testing"""
    total_reads = 0
    total_windows = 0
    
    print(f"Creating fast database from {zip_path} (max {max_reads} reads)...")
    
    with open(output_db, 'wb') as db, zipfile.ZipFile(zip_path, 'r') as z:
        fastq_files = [f for f in z.namelist() if f.endswith('.fastq')]
        
        for fname in fastq_files[:1]:  # Only first file for speed
            print(f"Processing {fname}...")
            with z.open(fname) as f:
                lines = []
                read_count = 0
                
                for line in f:
                    if total_reads >= max_reads:
                        break
                        
                    lines.append(line.decode().strip())
                    if len(lines) == 4:
                        res = process_read_simple(lines[1], lines[3])
                        if res:
                            phi, r = res
                            # Save Format: [NumWindows(I)][PhiData(half)][RData(half)]
                            db.write(struct.pack('<I', len(phi) // 8))  # Number of windows
                            db.write(phi.tobytes())
                            db.write(r.tobytes())
                            total_reads += 1
                            total_windows += len(phi) // 8
                            read_count += 1
                            
                            if total_reads % 100 == 0:
                                print(f"  Processed {total_reads} reads...")
                        
                        lines = []
                
                print(f"  Finished {fname}: {read_count} reads")
                
                if total_reads >= max_reads:
                    break
    
    db_size = os.path.getsize(output_db)
    print(f"\nFast Database Complete:")
    print(f"  Total Reads: {total_reads:,}")
    print(f"  Total Windows: {total_windows:,}")
    print(f"  Database Size: {db_size / 1024**2:.1f} MB")
    return total_reads, total_windows

class FastVRAMDB:
    def __init__(self, db_path):
        print(f"Loading fast database from {db_path}...")
        with open(db_path, 'rb') as f:
            self.raw_bytes = f.read()
        
        # Parse into VRAM tensors
        all_phi, all_r = [], []
        offset = 0
        total_reads = 0
        
        while offset < len(self.raw_bytes):
            if offset + 4 > len(self.raw_bytes):
                break
                
            n_win = struct.unpack_from('<I', self.raw_bytes, offset)[0]
            offset += 4
            
            if offset + n_win * 8 * 2 > len(self.raw_bytes):
                break
                
            # Read phi and r data
            phi = np.frombuffer(self.raw_bytes, dtype=np.float16, count=n_win*8, offset=offset)
            offset += n_win * 8 * 2
            r = np.frombuffer(self.raw_bytes, dtype=np.float16, count=n_win*8, offset=offset)
            offset += n_win * 8 * 2
            
            all_phi.append(phi.reshape(-1, 8))
            all_r.append(r.reshape(-1, 8))
            total_reads += 1
        
        print(f"Loading {total_reads} reads into VRAM...")
        self.phi_vram = cp.array(np.vstack(all_phi))
        self.r_vram = cp.array(np.vstack(all_r))
        
        vram_mb = (self.phi_vram.nbytes + self.r_vram.nbytes) / 1024**2
        print(f"VRAM Ready: {self.phi_vram.shape[0]} windows ({vram_mb:.1f} MB)")

    def search(self, motif, threshold=0.6):
        """Fast GPU search"""
        # Simple motif signature (no FWHT)
        z = [1.0 * np.exp(1j * PHASES[BASES[b]]) for b in motif[:8]]
        query_sig = cp.array(np.angle(z), dtype=cp.float16)
        
        # Vectorized similarity
        diff = self.phi_vram - query_sig
        scores = cp.mean(cp.cos(diff) * self.r_vram, axis=1)
        
        matches = cp.where(scores > threshold)[0]
        match_scores = scores[matches]
        
        # Sort by score
        sorted_indices = cp.argsort(match_scores)[::-1]
        return matches[sorted_indices], match_scores[sorted_indices]

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Create fast database
    reads, windows = convert_fast_subset('fastqfiles.zip', 'fast_test.polardb', max_reads=500)
    
    if reads > 0:
        print("\n" + "="*50)
        print("FAST VRAM DATABASE SEARCH")
        print("="*50)
        
        # Load and test
        db = FastVRAMDB('fast_test.polardb')
        
        # Test motifs
        test_motifs = ['GTCGCTCT', 'ATGCGACAG', 'GTGCCGTTC']
        
        for motif in test_motifs:
            print(f"\nSearching: {motif}")
            matches, scores = db.search(motif, threshold=0.5)
            
            if len(matches) > 0:
                print(f"  Found {len(matches)} matches")
                for i in range(min(3, len(matches))):
                    print(f"    Window {int(matches[i])}: {float(scores[i]):.4f}")
            else:
                print("  No matches found")
    else:
        print("No reads processed - check input file")
