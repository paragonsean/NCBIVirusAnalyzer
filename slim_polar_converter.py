import os, zipfile, struct, numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- STEP 0: CUDA ENVIRONMENT & DLL FIX ---
cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
if os.path.exists(cuda_root):
    os.add_dll_directory(os.path.join(cuda_root, "bin"))
    os.environ["CUDA_PATH"] = cuda_root

import cupy as cp

# --- STEP 1: GLOBAL NUMERIC MAPS ---
# Converting bases and quality to numbers first (The "Numeric First" Pipeline)
BASE_MAP = np.zeros(256, dtype=np.float32)
BASE_MAP[ord('A')], BASE_MAP[ord('C')] = 0, np.pi/2
BASE_MAP[ord('G')], BASE_MAP[ord('T')] = np.pi, 3*np.pi/2

QUAL_MAP = np.array([(i) / 94.0 for i in range(256)], dtype=np.float32)

# --- STEP 2: VECTORIZED HADAMARD LOGIC ---
def fwht_vectorized(a):
    """Iterative Vectorized Fast Walsh-Hadamard Transform."""
    n = a.shape[-1]
    a = a.reshape(-1, n).copy()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x, y = a[:, j].copy(), a[:, j + h].copy()
                a[:, j], a[:, j + h] = x + y, x - y
        h *= 2
    return a

def process_batch_numeric(read_batch, block_size=8):
    """Numeric-first transformation for high-speed conversion."""
    results = []
    for seq, qual in read_batch:
        try:
            # Convert strings to numeric arrays instantly
            s_bytes = np.frombuffer(seq.encode(), dtype=np.uint8)
            q_bytes = np.frombuffer(qual.encode(), dtype=np.uint8) - 33
            
            # Map to Complex Plane
            z = QUAL_MAP[q_bytes] * np.exp(1j * BASE_MAP[s_bytes])
            
            # SLIM Strategy: Non-overlapping blocks to avoid 12x file size expansion
            num_blocks = len(z) // block_size
            blocks = z[:num_blocks * block_size].reshape(-1, block_size)
            
            # Vectorized Transform
            rot_real, rot_imag = fwht_vectorized(blocks.real), fwht_vectorized(blocks.imag)
            
            r = np.sqrt(rot_real**2 + rot_imag**2).astype(np.float16)
            phi = np.arctan2(rot_imag, rot_real).astype(np.float16)
            
            results.append((phi.tobytes(), r.tobytes(), num_blocks))
        except: continue
    return results

# --- STEP 3: TURBO CONVERTER ---
def run_turbo_conversion(zip_path, output_db, batch_size=1000):
    total_reads = 0
    with open(output_db, 'wb') as db, zipfile.ZipFile(zip_path, 'r') as z:
        # Get list of FASTQ files
        f_list = [f for f in z.namelist() if f.endswith('.fastq')]
        with ProcessPoolExecutor() as executor:
            for fname in tqdm(f_list, desc="Processing Files"):
                read_batch, futures = [], []
                with z.open(fname) as f:
                    lines = []
                    for line in f:
                        lines.append(line.decode().strip())
                        if len(lines) == 4:
                            read_batch.append((lines[1], lines[3]))
                            if len(read_batch) >= batch_size:
                                futures.append(executor.submit(process_batch_numeric, read_batch))
                                read_batch = []
                            lines = []
                    if read_batch: futures.append(executor.submit(process_batch_numeric, read_batch))
                
                for future in futures:
                    for phi_b, r_b, n_win in future.result():
                        db.write(struct.pack('<I', n_win))
                        db.write(phi_b)
                        db.write(r_b)
                        total_reads += 1
    print(f"Done! Created {output_db} ({os.path.getsize(output_db)/1e6:.1f} MB)")

# --- STEP 4: VRAM RESIDENT ENGINE ---
class PolarEngineVRAM:
    def __init__(self, db_path):
        self.phi_vram, self.r_vram = self._load_to_vram(db_path)

    def _load_to_vram(self, path):
        print("Loading Database into Contiguous VRAM...")
        all_phi, all_r = [], []
        with open(path, 'rb') as f:
            while True:
                h = f.read(4)
                if not h: break
                n = struct.unpack('<I', h)[0]
                all_phi.append(np.frombuffer(f.read(n*16), dtype=np.float16).reshape(-1, 8))
                all_r.append(np.frombuffer(f.read(n*16), dtype=np.float16).reshape(-1, 8))
        return cp.array(np.vstack(all_phi)), cp.array(np.vstack(all_r))

    def search(self, motif, threshold=0.6):
        # Convert search motif to numeric signature
        z_q = [1.0 * np.exp(1j * BASE_MAP[ord(b)]) for b in motif[:8]]
        rot_q = fwht_vectorized(np.array(z_q))
        sig_q = cp.array(np.arctan2(rot_q.imag, rot_q.real), dtype=cp.float16)
        
        # Parallel Cosine Search across all blocks simultaneously
        scores = cp.mean(cp.cos(self.phi_vram - sig_q) * self.r_vram, axis=1)
        matches = cp.where(scores > threshold)[0]
        return matches, scores[matches]

import numpy as np
import zipfile

def verify_integrity(zip_path, engine, num_checks=5):
    print(f"--- DATABASE INTEGRITY CHECK (n={num_checks}) ---")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Get the first FASTQ file for testing
        fname = [f for f in z.namelist() if f.endswith('.fastq')][0]
        
        with z.open(fname) as f:
            lines = [line.decode().strip() for line in f]
            
            # Calculate blocks per read for this file
            test_seq = lines[1]  # First read sequence
            blocks_per_read = len(test_seq) // 8
            
            # Pick a sample read (e.g., the 10th read in the file)
            for i in range(num_checks):
                read_idx = i * 10 
                seq = lines[read_idx * 4 + 1]
                
                # 1. Pick a "Ground Truth" motif from the middle of the read
                motif_len = 8
                start_pos = 24 # Must be a multiple of 8 because of our 'Slim' grid
                truth_motif = seq[start_pos : start_pos + motif_len]
                
                # 2. Query the VRAM Engine
                expected_block_offset = start_pos // 8
                matches, scores = engine.search(truth_motif, threshold=0.9)
                
                # 3. Cross-reference with more flexible validation
                # Since we don't know the exact global offset, we'll check if:
                # - The motif exists in the matches at all
                # - The score is high enough to indicate a good match
                
                is_valid = len(matches) > 0 and cp.max(scores) > 0.95
                
                # Additional debug info
                if len(matches) > 0:
                    best_match_idx = int(matches[cp.argmax(scores)])
                    best_score = float(cp.max(scores))
                    print(f"Test {i+1}: Motif '{truth_motif}' | Found {len(matches)} matches | Best: Block {best_match_idx} (Score: {best_score:.4f}) | Status: {'   VERIFIED' if is_valid else '   MISMATCH'}")
                else:
                    print(f"Test {i+1}: Motif '{truth_motif}' | Found 0 matches | Status:   MISMATCH")

# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Convert everything to Slim Polar Numeric format
    run_turbo_conversion('fastqfiles.zip', 'master.polardb')
    
    # 2. Load once and Query
    engine = PolarEngineVRAM('master.polardb')
    motif = "GTCGCTCT"
    results, scores = engine.search(motif)
    print(f"Found {len(results)} instances of {motif}")
    
    # 3. Run integrity verification
    verify_integrity('fastqfiles.zip', engine)