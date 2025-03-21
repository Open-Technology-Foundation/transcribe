#!/bin/bash
# Benchmark script to compare performance of different parallel processing configurations

set -euo pipefail

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OUTPUT_DIR="/tmp/transcribe_benchmark"
MAX_WORKERS=$(nproc)

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to print section header
print_header() {
  echo -e "\n${BLUE}==== $1 ====${NC}\n"
}

# Function to run a benchmark test
run_benchmark() {
  local name="$1"
  local audio_file="$2"
  local cmd_args="$3"
  local output_file="${OUTPUT_DIR}/${name// /_}.txt"
  
  echo -e "${YELLOW}Running test: ${name}${NC}"
  echo -e "Command: python3 -m transcribe_pkg ${audio_file} ${cmd_args} -o ${output_file}"
  
  # Measure time
  start_time=$(date +%s.%N)
  python3 -m transcribe_pkg "$audio_file" $cmd_args -o "$output_file" -v
  end_time=$(date +%s.%N)
  
  # Calculate elapsed time
  elapsed=$(echo "$end_time - $start_time" | bc)
  elapsed_rounded=$(printf "%.2f" "$elapsed")
  
  echo -e "${GREEN}Test completed in ${elapsed_rounded} seconds${NC}"
  echo "-------------------------------------------------"
  echo ""
  
  # Return elapsed time
  echo "$elapsed_rounded"
}

# Check for audio file argument
if [ $# -lt 1 ]; then
  echo -e "${RED}Error: No audio file provided${NC}"
  echo "Usage: $0 <audio_file> [--skip-large]"
  exit 1
fi

AUDIO_FILE="$1"
SKIP_LARGE="${2:-}"

# Verify audio file exists
if [ ! -f "$AUDIO_FILE" ]; then
  echo -e "${RED}Error: Audio file not found: $AUDIO_FILE${NC}"
  exit 1
fi

print_header "Parallel Processing Benchmark"
echo "Audio file: $AUDIO_FILE"
echo "Max workers available: $MAX_WORKERS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run small benchmarks
print_header "Basic Benchmarks (No Post-Processing)"

# Sequential (baseline)
seq_time=$(run_benchmark "Sequential (no post)" "$AUDIO_FILE" "-P")

# Parallel transcription only
par_time=$(run_benchmark "Parallel transcription (no post)" "$AUDIO_FILE" "-P --parallel")

# Skip larger tests if requested
if [ "$SKIP_LARGE" = "--skip-large" ]; then
  echo -e "${YELLOW}Skipping full processing benchmarks${NC}"
  exit 0
fi

print_header "Full Processing Benchmarks (With Post-Processing)"

# Sequential with post-processing
seq_post_time=$(run_benchmark "Sequential with post" "$AUDIO_FILE" "")

# Parallel transcription with sequential post-processing
par_seq_post_time=$(run_benchmark "Parallel trans, sequential post" "$AUDIO_FILE" "--parallel --max-parallel-workers 1")

# Fully parallel (transcription and post-processing)
full_par_time=$(run_benchmark "Fully parallel" "$AUDIO_FILE" "--parallel --max-parallel-workers 4")

# Use the wrapper script
wrapper_time=$(run_benchmark "Optimized wrapper" "$AUDIO_FILE" "--parallel --content-aware --cache")

# Print summary
print_header "Performance Summary"
echo -e "${GREEN}No Post-Processing:${NC}"
echo "  Sequential: ${seq_time}s"
echo "  Parallel transcription: ${par_time}s"
echo -e "  Speedup: $(echo "scale=2; $seq_time / $par_time" | bc)x"
echo ""

echo -e "${GREEN}With Post-Processing:${NC}"
echo "  Sequential: ${seq_post_time}s"
echo "  Parallel transcription only: ${par_seq_post_time}s"
echo "  Fully parallel: ${full_par_time}s"
echo "  Optimized wrapper: ${wrapper_time}s"
echo ""

echo -e "${GREEN}Overall Speedup:${NC}"
echo -e "  $(echo "scale=2; $seq_post_time / $full_par_time" | bc)x faster with parallel processing"
echo ""

echo -e "${BLUE}Benchmark complete! Results saved to ${OUTPUT_DIR}${NC}"

#fin