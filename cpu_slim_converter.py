import itertools
import os

def compress_polar_dna(sequence: str) -> str:
    """
    Compresses a DNA sequence using the 13-limit ASCII chunking method.
    """
    # Added 'N' for unknown bases commonly found in FASTA files
    offsets = {
        'A': 97,  # Starts at 'a' (97-109)
        'C': 65,  # Starts at 'A' (65-77)
        'T': 78,  # Starts at 'N' (78-90)
        'G': 110, # Starts at 'n' (110-122)
        'N': 33   # Starts at '!' (33-45) 
    }
    max_chunk = 13
    compressed_string = []

    for base, group in itertools.groupby(sequence.upper()):
        count = sum(1 for _ in group)
        
        # Skip newline characters or spaces that might sneak in
        if base not in offsets:
            continue
            
        offset = offsets[base]
        full_chunks = count // max_chunk
        remainder = count % max_chunk

        if full_chunks > 0:
            max_char = chr(offset + (max_chunk - 1))
            compressed_string.append(max_char * full_chunks)

        if remainder > 0:
            rem_char = chr(offset + (remainder - 1))
            compressed_string.append(rem_char)

    return "".join(compressed_string)

def convert_fasta_to_polar(input_filepath: str, output_filepath: str):
    """
    Reads a FASTA file, preserves headers, and compresses the sequences.
    """
    print(f"Opening {input_filepath}...")
    
    with open(input_filepath, 'r') as infile, open(output_filepath, 'w') as outfile:
        header = ""
        sequence_buffer = []
        sequence_count = 0
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(">"):
                # If we have a buffered sequence, compress and write it
                if header:
                    full_seq = "".join(sequence_buffer)
                    compressed_seq = compress_polar_dna(full_seq)
                    outfile.write(f"{header}\n{compressed_seq}\n")
                    sequence_count += 1
                
                # Start the new sequence
                header = line
                sequence_buffer = [] 
            else:
                # Accumulate sequence lines
                sequence_buffer.append(line)
        
        # Catch and process the final sequence in the file
        if header and sequence_buffer:
            full_seq = "".join(sequence_buffer)
            compressed_seq = compress_polar_dna(full_seq)
            outfile.write(f"{header}\n{compressed_seq}\n")
            sequence_count += 1

    # Calculate file size savings
    original_size = os.path.getsize(input_filepath) / (1024 * 1024)
    new_size = os.path.getsize(output_filepath) / (1024 * 1024)
    
    print(f"\n--- Conversion Complete ---")
    print(f"Sequences Processed: {sequence_count}")
    print(f"Original Size: {original_size:.2f} MB")
    print(f"Compressed Size: {new_size:.2f} MB")
    print(f"File saved to: {output_filepath}")

if __name__ == "__main__":
    # Point this to your uploaded file
    input_file = "influenzaA.fasta"
    output_file = "sequences.polar"
    
    if os.path.exists(input_file):
        convert_fasta_to_polar(input_file, output_file)
    else:
        print(f"Error: Could not find {input_file} in the current directory.")