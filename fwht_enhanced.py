import os
import numpy as np

# --- STEP 0: CUDA INITIALIZATION ---
cuda_root = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
if os.path.exists(cuda_root):
    os.add_dll_directory(os.path.join(cuda_root, "bin"))
    os.environ["CUDA_PATH"] = cuda_root

import cupy as cp

# --- STEP 1: ENHANCED POLAR ENGINE ---
BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

def fwht(a):
    n = len(a)
    if n == 1: return a
    h = n // 2
    l, r = fwht(a[0:h]), fwht(a[h:n])
    return np.concatenate([l + r, l - r])

def fastq_to_sliding_vram(seq, qual_str, block_size=8):
    """Creates overlapping blocks so we can find motifs at any offset."""
    z = []
    for b, q in zip(seq, qual_str):
        if b not in BASES: continue
        m = (ord(q) - 33 + 1) / 94.0
        z.append(m * np.exp(1j * PHASES[BASES[b]]))
    
    z = np.array(z)
    # Create sliding windows: [0:8], [1:9], [2:10]...
    num_windows = len(z) - block_size + 1
    if num_windows <= 0: return None, None
    
    # Efficiently create the overlapping blocks
    shape = (num_windows, block_size)
    strides = (z.strides[0], z.strides[0])
    windows = np.lib.stride_tricks.as_strided(z, shape=shape, strides=strides)
    
    phi_list, r_list = [], []
    for window in windows:
        rot_real, rot_imag = fwht(window.real), fwht(window.imag)
        r_list.append(np.sqrt(rot_real**2 + rot_imag**2))
        phi_list.append(np.arctan2(rot_imag, rot_real))
        
    return cp.array(phi_list, dtype=cp.float32), cp.array(r_list, dtype=cp.float32)

def get_motif_sig(motif):
    z = [1.0 * np.exp(1j * PHASES[BASES[b]]) for b in motif]
    rot_real, rot_imag = fwht(np.real(z)), fwht(np.imag(z))
    return cp.array(np.arctan2(rot_imag, rot_real), dtype=cp.float32)

# --- STEP 2: MULTI-READ DATA ---
# Let's use the actual sequences you provided earlier
fastq_data = [
    ("GTGTGCCGTTCGTTTTCGAGAAGTTTGGGTGTTTATGGATCGCCTACCGTGACTTCGAACAAACCAAGTTACGTACTTGCCTGTCGCTCTGTCTTCAGTCTTGGTCCATTGTTTCGAAGACCAGCGCGATCAGTATGCGACAGGCGGTGTATTAATATCAGCACCAACAAAGCGTAACGGTTTGTTCCCTGGGGGT",
     "%+&+%##&**%#,.-+&%'$$%$$&'&+&2+124,+&(#')''(%###$'$%%(/&&$4<24.25()72/%-&0+,..144;3-407=63&9;8*+(#)-&)/3%%6-.1,=?*+$$$1(..,),',+*((#'(%$%$.),$/-5$++%%&$&%-,-16.48++6**2&&%'+#$&&)17=>/-87$+$%$'')*("),
    ("GTGGTGGTGTACTTCGTTCCCGTTTCTGTTTGGGTGTTATGATCGTCGCCTACCGTGACAGGGAACAAACCAAGTTACGTTTTCTGATGGTGCTGATATTACCGCACCGCCCGTCGCTACTACCGATTGTGGCGGACAGCGCGCTGCTTAACGTTGGCTACGGCAACTGCGCGCAGCTCCCGGCGCTTCGCCTGTGGGCTCGACCCGTCGAACACGGACCAGAGTCTGAAGATGAGCAGGCGAAATTGCGTAGCGGTTTACGTTCCCTGAAGTC",
     "%$%$$#%'$$$#$%./88%(+)+.&&%($%)')-2+))%$)$#$$#)%%%.+)++-*(&$'+)*06+4/;>23914+-+)*?62,1',$&,&&**%)$''#'($&.02:=:1+.=$$6755./*2$$(:0/5D3:+%'-36/'#%###$('%$&+%'$'#-05;4'%.%&($&&$+01$+'$-40&''%(%&&&)(*8;3*31;8A31/.-%----4C844*#;7;.62<52,('&2-.22&()$$*%1.(.%.&,22'\"%+.1/4*))$%&%%")
]

# --- STEP 3: SEARCH ALL READS ---
target = "GTCGCTCT"
query_sig = get_motif_sig(target)

print(f"Target Motif: {target}")
print("-" * 40)

for i, (seq, qual) in enumerate(fastq_data):
    phi_vram, r_vram = fastq_to_sliding_vram(seq, qual)
    
    if phi_vram is not None:
        diff = phi_vram - query_sig
        # Cosine similarity weighted by quality magnitude
        scores = cp.mean(cp.cos(diff) * r_vram, axis=1)
        
        # High threshold for precision
        matches = cp.where(scores > 0.7)[0]
        
        if len(matches) > 0:
            best_idx = int(matches[cp.argmax(scores[matches])])
            print(f"Read {i}: Found {len(matches)} matches. Top score at base {best_idx}: {float(scores[best_idx]):.4f}")
            print(f"   Context: {seq[best_idx : best_idx+len(target)]}")
        else:
            print(f"Read {i}: No matches found above threshold")
    else:
        print(f"Read {i}: Sequence too short for analysis")
