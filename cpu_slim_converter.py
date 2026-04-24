import os
import zipfile
import struct
import numpy as np
from tqdm import tqdm
import time

# --- CORE POLAR LOGIC ---
BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

def fwht(a):
    """Fast Walsh-Hadamard Transform (Recursive)"""
    n = len(a)
    if n == 1: return a
    h = n // 2
    l, r = fwht(a[0:h]), fwht(a[h:n])
    return np.concatenate([l + r, l - r])

def process_read_slim(seq, qual, block_size=8):
    """
    SLIM VERSION: Only saves non-overlapping blocks to avoid 12x expansion.
    """
    z = []
    for b, q in zip(seq, qual):
        if b not in BASES: continue
        m = (ord(q) - 33 + 1) / 94.0
        z.append(m * np.exp(1j * PHASES[BASES[b]]))
    
    if len(z) < block_size: return None
    z = np.array(z)
    
    # --- NO SLIDING STRIDE HERE ---
    # We step by block_size (8) instead of 1.
    num_blocks = len(z) // block_size
    z_trimmed = z[:num_blocks * block_size]
    blocks = z_trimmed.reshape(-1, block_size)
    
    phi_list, r_list = [], []
    for w in blocks:
        # Standard Hadamard rotation
        rr, ri = fwht(w.real), fwht(w.imag)
        r_list.append(np.sqrt(rr**2 + ri**2))
        phi_list.append(np.arctan2(ri, rr))
    
    return np.array(phi_list, dtype=np.float16), np.array(r_list, dtype=np.float16)

def convert_to_slim_db(zip_path, output_db, max_reads=10000):
    total_reads = 0
    total_blocks = 0
    
    print(f"{'='*60}")
    print("CPU SLIM POLAR DATABASE CONVERSION")
    print(f"{'='*60}")
    print(f"Max Reads: {max_reads:,}")
    print(f"Output: {output_db}")
    print("Mode: Non-overlapping blocks (no sliding windows)")
    print("Backend: CPU-only (no CUDA required)")
    
    with open(output_db, 'wb') as db, zipfile.ZipFile(zip_path, 'r') as z:
        fastq_files = [f for f in z.namelist() if f.endswith('.fastq')]
        
        file_pbar = tqdm(fastq_files, desc="Converting to Slim DB", unit="file")
        
        for fname in file_pbar:
            if total_reads >= max_reads:
                file_pbar.set_description(f"Reached {max_reads} reads limit")
                break
                
            short_name = fname.split('/')[-1] if '/' in fname else fname
            file_pbar.set_description(f"Processing {short_name}")
            
            with z.open(fname) as f:
                lines = []
                file_reads = 0
                
                for line in f:
                    if total_reads >= max_reads:
                        break
                        
                    lines.append(line.decode().strip())
                    if len(lines) == 4:
                        res = process_read_slim(lines[1], lines[3])
                        if res:
                            phi, r = res
                            # Save with proper format: [num_blocks][phi_data][r_data]
                            db.write(struct.pack('<I', len(phi)))
                            db.write(phi.tobytes())
                            db.write(r.tobytes())
                            total_reads += 1
                            total_blocks += len(phi)
                            
                            # Update progress every 1000 reads
                            if total_reads % 1000 == 0:
                                file_pbar.set_postfix({
                                    'reads': f"{total_reads:,}",
                                    'blocks': f"{total_blocks:,}"
                                })
                        lines = []
                        file_reads += 1
            
            print(f"  Completed {file_reads} reads from {short_name}")
    
    final_size = os.path.getsize(output_db) / 1024**2
    
    print(f"\n{'='*60}")
    print("CPU SLIM CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total Reads: {total_reads:,}")
    print(f"Total Blocks: {total_blocks:,}")
    print(f"Database Size: {final_size:.1f} MB")
    print(f"Blocks per Read: {total_blocks / total_reads:.1f}")
    print(f"Output File: {output_db}")
    print(f"\nSize Comparison: {final_size:.1f} MB (vs ~9000 MB with sliding windows)")
    
    return total_reads, total_blocks

class CPUSlimDB:
    def __init__(self, db_path):
        print(f"\n{'='*60}")
        print("LOADING CPU SLIM DATABASE")
        print(f"{'='*60}")
        print(f"Database: {db_path}")
        print("Backend: CPU-only (no CUDA required)")
        
        self.db_path = db_path
        self.offsets = []
        self.window_counts = []
        
        # Build index
        db_size = os.path.getsize(db_path)
        print("Building database index...")
        
        with open(db_path, 'rb') as f:
            with tqdm(total=db_size, unit='B', unit_scale=True, desc="Indexing Database") as pbar:
                offset = 0
                while True:
                    header = f.read(4)
                    if not header: break
                    num_windows = struct.unpack('<I', header)[0]
                    self.window_counts.append(num_windows)
                    self.offsets.append(offset)
                    
                    record_size = 4 + num_windows * 8 * 2 
                    f.seek(record_size, 1)
                    offset += (4 + record_size)
                    pbar.update(4 + record_size)
        
        print(f"Indexed {len(self.offsets)} reads")
        
        # Load entire slim database into memory
        self._load_memory()

    def _load_memory(self):
        """Load entire slim database into CPU memory"""
        load_start = time.time()
        
        # Read all data with error handling
        all_phi, all_r = [], []
        corrupted_records = 0
        
        print("Loading slim database into memory...")
        load_pbar = tqdm(zip(self.offsets, self.window_counts), 
                        total=len(self.offsets), 
                        desc="Loading to Memory", unit="records")
        
        with open(self.db_path, 'rb') as f:
            for i, (offset, num_windows) in enumerate(load_pbar):
                try:
                    f.seek(offset)
                    header = f.read(4)
                    if len(header) != 4:
                        print(f"  Warning: Incomplete header at record {i}")
                        corrupted_records += 1
                        continue
                    
                    n_win = struct.unpack('<I', header)[0]
                    
                    # Read phi data
                    phi_bytes = f.read(n_win * 8 * 2)
                    if len(phi_bytes) != n_win * 8 * 2:
                        print(f"  Warning: Incomplete phi data at record {i}")
                        corrupted_records += 1
                        continue
                    
                    # Read r data
                    r_bytes = f.read(n_win * 8 * 2)
                    if len(r_bytes) != n_win * 8 * 2:
                        print(f"  Warning: Incomplete r data at record {i}")
                        corrupted_records += 1
                        continue
                    
                    # Verify data size before reshaping
                    phi_array = np.frombuffer(phi_bytes, dtype=np.float16)
                    r_array = np.frombuffer(r_bytes, dtype=np.float16)
                    
                    if len(phi_array) % 8 != 0 or len(r_array) % 8 != 0:
                        print(f"  Warning: Invalid data size at record {i}")
                        corrupted_records += 1
                        continue
                    
                    phi = phi_array.reshape(-1, 8)
                    r = r_array.reshape(-1, 8)
                    
                    all_phi.append(phi)
                    all_r.append(r)
                    
                except Exception as e:
                    print(f"  Warning: Error loading record {i}: {e}")
                    corrupted_records += 1
                    continue
        
        load_pbar.close()
        
        if not all_phi:
            print(f"  Error: No valid data loaded from database")
            return
        
        # Load into memory
        try:
            print("Concatenating into memory arrays...")
            concat_start = time.time()
            self.phi_memory = np.vstack(all_phi)
            self.r_memory = np.vstack(all_r)
            concat_time = time.time() - concat_start
            
            memory_mb = (self.phi_memory.nbytes + self.r_memory.nbytes) / 1024**2
            total_time = time.time() - load_start
            
            print(f"Memory Loading Complete:")
            print(f"  Blocks: {self.phi_memory.shape[0]:,}")
            print(f"  Memory Used: {memory_mb:.1f} MB")
            print(f"  Load Time: {total_time:.1f}s (concat: {concat_time:.1f}s)")
            print(f"  Valid Records: {len(all_phi)}/{len(self.offsets)}")
            if corrupted_records > 0:
                print(f"  Corrupted Records: {corrupted_records}")
            
        except Exception as e:
            print(f"  Error: Failed to load database into memory: {e}")
            return

    def search(self, motif, threshold=0.6, max_results=100):
        """Fast CPU search on slim database"""
        print(f"\nSearching for motif: '{motif}'")
        
        # Create motif signature
        z = [1.0 * np.exp(1j * PHASES[BASES[b]]) for b in motif[:8]]
        rr, ri = fwht(np.real(z)), fwht(np.imag(z))
        query_sig = np.arctan2(ri, rr)
        
        # Vectorized similarity calculation
        search_start = time.time()
        
        # Calculate similarity for all blocks
        similarities = []
        for i in range(self.phi_memory.shape[0]):
            diff = self.phi_memory[i] - query_sig
            score = np.mean(np.cos(diff) * self.r_memory[i])
            similarities.append(score)
        
        similarities = np.array(similarities)
        
        # Find matches
        matches = np.where(similarities > threshold)[0]
        match_scores = similarities[matches]
        
        # Sort by score
        sorted_indices = np.argsort(match_scores)[::-1]
        top_matches = matches[sorted_indices[:max_results]]
        top_scores = match_scores[sorted_indices[:max_results]]
        
        search_time = time.time() - search_start
        
        print(f"CPU Search Complete:")
        print(f"  Matches Found: {len(matches)} (above {threshold})")
        print(f"  Search Time: {search_time:.6f}s")
        if search_time > 0:
            search_rate = self.phi_memory.shape[0] / search_time
            print(f"  Search Rate: {search_rate:.0f} blocks/s")
        else:
            print(f"  Search Rate: Instant (sub-microsecond)")
        
        return top_matches, top_scores

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Create CPU slim database
    reads, blocks = convert_to_slim_db('fastqfiles.zip', 'cpu_slim_genome.polardb', max_reads=5000)
    
    if reads > 0:
        print(f"\n{'='*60}")
        print("CPU SLIM DATABASE SEARCH DEMO")
        print(f"{'='*60}")
        
        # Load database
        db = CPUSlimDB('cpu_slim_genome.polardb')
        
        # Test motifs
        test_motifs = ['GTCGCTCT', 'ATGCGACAG', 'GTGCCGTTC', 'CGTGACTTC', 'AACAAACCA']
        
        for motif in test_motifs:
            matches, scores = db.search(motif, threshold=0.5)
            
            if len(matches) > 0:
                print(f"\nTop 5 matches for '{motif}':")
                for i in range(min(5, len(matches))):
                    print(f"  Block {int(matches[i]):6d}: Score {float(scores[i]):.4f}")
            else:
                print(f"\nNo matches found for '{motif}' above threshold")
    else:
        print("\nNo reads processed - check input file")
