#!/bin/bash

# Update package list and install required tools if not already installed
#sudo apt update && sudo apt install -y zstd coreutils parallel

PREFIX=enwiki-20150205-pages-articles-chunk-
FILENAME=enwiki-20150205-pages-articles.xml

echo "Splitting $FILENAME into 2GB chunks with " \
     "numeric suffixes and .xml extension..."

# Split the XML file into 2GB chunks with numeric suffixes and .xml extension
split -b 2G --numeric-suffixes=4 \
            --additional-suffix=.xml \
            "$FILENAME" \
            "$PREFIX"

# Get the number of CPU cores for parallel processing
num_cores=$(nproc)

echo "Compressing $FILENAME chunks with zstd using $num_cores threads..."

# Compress the chunks in parallel, using one thread per zstd job.
time parallel -j "$num_cores" 'zstd -19 --ultra -T1 {}' ::: ${PREFIX}*.xml

echo "Done!"

# Optionally remove the uncompressed chunks
# rm "${PREFIX}"*.xml
