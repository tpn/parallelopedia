#!/bin/bash

# Define filenames and prefix.
PREFIX=enwiki-20150205-pages-articles-chunk-
OUTPUT_FILENAME=enwiki-20150205-pages-articles.xml
ORIGINAL_FILENAME="$OUTPUT_FILENAME"

# If OUTPUT_FILENAME already exists, alter the name to avoid overwriting.
if [ -f "$OUTPUT_FILENAME" ]; then
    echo "Output file $OUTPUT_FILENAME already exists.  Changing output " \
         "filename to $OUTPUT_FILENAME.new."
    OUTPUT_FILENAME="$OUTPUT_FILENAME.new"
fi

echo "Decompressing chunks with zstd using $(nproc) threads..."

# Get the number of CPU cores for parallel processing.
num_cores=$(nproc)

# Decompress the chunks in parallel, using one thread per zstd job.
time parallel -j "$num_cores" 'zstd -d -T1 {}' ::: ${PREFIX}*.xml.zst

echo "Joining decompressed chunks into $OUTPUT_FILENAME..."

# Concatenate all decompressed chunks into the final output file
cat ${PREFIX}*.xml > "$OUTPUT_FILENAME"

echo "Verification: Checking file sizes..."
original_size=$(ls -lh "$OUTPUT_FILENAME" | awk '{print $5}')
echo "Reconstructed file size: $original_size"

echo "Done!"

# Optionally remove the decompressed chunks after joining
# rm "${PREFIX}"*.xml

