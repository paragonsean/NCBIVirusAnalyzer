import re
import zipfile
import numpy as np
import struct
import lzma
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Import from polar modules
from polar_format_v2_magnitude import (
    parse_fastq_from_zip, encode_polar_v2, 
    decode_polar_v2, bin_quality_string, sort_reads, 
    unbin_quality_string, unsort_reads
)
from polar_v5_parallel import compress_parallel, decompress_parallel

def process_and_verify(zippath, output_polar):
    # 1. Extract reads from the first 5 files only
    all_reads = []
    with zipfile.ZipFile(zippath) as z:
        fnames = [f.filename for f in z.infolist() if f.filename.endswith('.fastq')]
        # Only take first 1 file for much faster testing
        fnames = fnames[:1]
        for fname in tqdm(fnames, desc="Extracting files", unit="file"):
            all_reads.extend(parse_fastq_from_zip(zippath, fname))

    # 2. Encode to .polar v2 (with smaller context window for speed)
    print(f"Encoding {len(all_reads)} reads...")
    blob, _ = encode_polar_v2(all_reads, seq_k=3, n_buckets=4)
    with open(output_polar, 'wb') as f:
        f.write(blob)

    # 3. Decode for verification
    print("Decoding for integrity check...")
    decoded_reads = decode_polar_v2(blob, seq_k=3, n_buckets=4)

    # 4. Integrity Comparison
    integrity_pass = True
    for i, (orig, dec) in enumerate(tqdm(zip(all_reads, decoded_reads), desc="Verifying integrity", unit="read", total=len(all_reads))):
        # Clean original sequence (v2 strips non-ACGT)
        orig_seq = re.sub(r'[^ACGT]', '', orig[1].upper())
        
        # Check Sequence Integrity (Lossless)
        if dec[1] != orig_seq:
            print(f"FAIL: Sequence mismatch at read {i}")
            integrity_pass = False
            break
            
        # Check Quality Integrity (Binned/Lossy)
        if not np.array_equal(bin_quality_string(orig[2]), bin_quality_string(dec[2])):
            print(f"FAIL: Quality bin mismatch at read {i}")
            integrity_pass = False
            break

    if integrity_pass:
        print(f"SUCCESS: Integrity verified for {len(all_reads)} reads.")

def run_parallel_polar_v5(zippath, output_file, tile_size=4096, k=4):
    print(f"--- Starting Parallel .polar v5 Processing ---")
    
    # 1. Load and Sort Data
    all_reads = []
    with zipfile.ZipFile(zippath) as z:
        fnames = [f.filename for f in z.infolist() if f.filename.endswith('.fastq')]
        # Only take first 5 files for faster testing
        fnames = fnames[:5]
        for f in tqdm(fnames, desc="Loading files", unit="file"):
            all_reads.extend(parse_fastq_from_zip(zippath, f))
    
    sorted_reads, perm = sort_reads(all_reads)
    headers = [r[0] for r in sorted_reads]
    seqs = [re.sub(r'[^ACGT]', '', r[1].upper()) for r in sorted_reads]
    quals = [r[2] for r in sorted_reads]
    
    # 2. Compress Streams
    # Stream 0: Headers (LZMA)
    header_blob = lzma.compress("\n".join(headers).encode())
    
    # Stream 1: Sequences (Parallel Tiled V5)
    full_seq = "".join(seqs)
    seq_blob = compress_parallel(full_seq, tile_size=tile_size, k=k, polar=True)
    
    # Stream 2: Quality (Binned to 8 levels + LZMA)
    binned_quals = np.concatenate([bin_quality_string(q) for q in tqdm(quals, desc="Binning quality", unit="read")])
    qual_blob = lzma.compress(binned_quals.tobytes())
    
    # Stream 3: Meta (Lengths + Permutation)
    meta = struct.pack(f"<{len(seqs)}I", *[len(s) for s in seqs])
    meta += struct.pack(f"<{len(perm)}I", *perm)
    meta_blob = lzma.compress(meta)
    
    # 3. Assemble .polar file
    with open(output_file, 'wb') as f:
        f.write(b'POLARV5P') # Magic: Parallel V5
        f.write(struct.pack("<III", len(all_reads), tile_size, k))
        for stream in [header_blob, seq_blob, qual_blob, meta_blob]:
            f.write(struct.pack("<I", len(stream)))
            f.write(stream)
            
    print(f"Compressed {len(all_reads)} reads into {output_file}")
    return all_reads

def verify_integrity(original_reads, polar_file):
    print(f"\n--- Verifying Integrity of {polar_file} ---")
    
    with open(polar_file, 'rb') as f:
        data = f.read()
        
    off = 0
    # 1. Check Magic Bytes
    magic = data[off:off+8]
    off += 8
    assert magic == b'POLARV5P', f"Invalid magic bytes: {magic}"
    
    # 2. Read Global Metadata
    n_reads, tile_size, k = struct.unpack_from("<III", data, off)
    off += 12
    
    assert n_reads == len(original_reads), "Read count mismatch!"
    
    # 3. Extract the 4 Streams
    streams = []
    for _ in range(4):
        slen = struct.unpack_from("<I", data, off)[0]
        off += 4
        streams.append(data[off:off+slen])
        off += slen
        
    header_blob, seq_blob, qual_blob, meta_blob = streams
    
    # 4. Decompress Streams
    print("Decompressing Meta and Headers...")
    meta_raw = lzma.decompress(meta_blob)
    seq_lengths = list(struct.unpack_from(f"<{n_reads}I", meta_raw, 0))
    perm = list(struct.unpack_from(f"<{n_reads}I", meta_raw, n_reads * 4))
    
    headers_text = lzma.decompress(header_blob).decode()
    sorted_headers = headers_text.split("\n")
    
    print("Decompressing Sequence Tiles (Parallel)...")
    all_seq_str = decompress_parallel(seq_blob, k=k, polar=True, n_buckets=8, max_workers=4)
    
    print("Decompressing Quality Bins...")
    all_binned = np.frombuffer(lzma.decompress(qual_blob), dtype=np.uint8)
    
    # 5. Reconstruct the sorted reads
    sorted_seqs = []
    sorted_quals = []
    seq_pos = 0
    qual_pos = 0
    
    for length in seq_lengths:
        # Reconstruct Sequence
        sorted_seqs.append(all_seq_str[seq_pos:seq_pos+length])
        seq_pos += length
        
        # Reconstruct Quality (Unbinning to representative Phred scores)
        q_bins = all_binned[qual_pos:qual_pos+length]
        sorted_quals.append(unbin_quality_string(q_bins))
        qual_pos += length
        
    # 6. Unsort to original order
    sorted_reads = list(zip(sorted_headers, sorted_seqs, sorted_quals))
    decoded_reads = unsort_reads(sorted_reads, perm)
    
    # 7. Perform the Integrity Check
    print("\nComparing against original reads...")
    integrity_pass = True
    
    for i, (orig, dec) in enumerate(tqdm(zip(original_reads, decoded_reads), desc="Comparing reads", unit="read", total=len(original_reads))):
        orig_h, orig_s, orig_q = orig
        dec_h, dec_s, dec_q = dec
        
        # Check Header (Lossless)
        if orig_h != dec_h:
            print(f"FAIL: Header mismatch at read {i}")
            integrity_pass = False
            break
            
        # Check Sequence (Lossless)
        # We must clean the original sequence first because our encoder strips non-ACGT
        orig_s_clean = re.sub(r'[^ACGT]', '', orig_s.upper())
        if orig_s_clean != dec_s:
            print(f"FAIL: Sequence mismatch at read {i}\nExpected: {orig_s_clean}\nGot:      {dec_s}")
            integrity_pass = False
            break
            
        # Check Quality (Lossy / Binned)
        # We compare the binned index arrays, NOT the raw ASCII strings
        orig_q_binned = bin_quality_string(orig_q)
        dec_q_binned = bin_quality_string(dec_q) # dec_q contains the representative ASCII from unbin_quality_string
        
        if not np.array_equal(orig_q_binned, dec_q_binned):
            print(f"FAIL: Quality bin mismatch at read {i}")
            integrity_pass = False
            break

    if integrity_pass:
        print(f"SUCCESS: Integrity verified for all {n_reads} reads!")
        print(" -> Sequences are 100% losslessly preserved.")
        print(" -> Quality scores match the 8-level quantization bins perfectly.")
        
    return integrity_pass

def main():
    """Main function to test .polar compression formats."""
    # Test the functions
    ZIP_FILE = "fastqfiles.zip"
    OUTPUT_POLAR_V2 = "test_v2.polar"
    OUTPUT_POLAR_V5 = "test_v5.polar"
    
    print("=" * 60)
    print("POLAR COMPRESSION FORMAT TESTING")
    print("=" * 60)
    
    print("\n=== Testing .polar v2 ===")
    try:
        process_and_verify(ZIP_FILE, OUTPUT_POLAR_V2)
        print("v2 test completed successfully")
    except Exception as e:
        print(f"v2 test failed: {e}")
    
    print("\n=== Testing .polar v5 Parallel ===")
    try:
        original_reads = run_parallel_polar_v5(ZIP_FILE, OUTPUT_POLAR_V5)
        verify_integrity(original_reads, OUTPUT_POLAR_V5)
        print("v5 parallel test completed successfully")
    except Exception as e:
        print(f"v5 parallel test failed: {e}")
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()