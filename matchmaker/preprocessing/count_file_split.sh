# 
# returns the batch count (that the multiprocess generators will produce)
#

# arg 1: split_files
# arg 2: batch size

if [ $# -ne 2 ]
  then
    echo "Usage (bash): ./count_file_split.sh <split_files> <batch_size>"
    exit 1
fi

base_file=$1
batch_size=$2

total_batch_number=0
total_split_lines=0

for file in "$base_file"*; do
    
    file_length=$(wc -l < "${file}")
    batch_count=$((($file_length+$batch_size-1)/$batch_size)) # round up division, ref: https://stackoverflow.com/a/2395027
    
    echo $file" lines: "$file_length" batches: "$batch_count
    
    total_batch_number=$((total_batch_number+batch_count))
    total_split_lines=$((total_split_lines+file_length))

done

echo "--------------------------------------------------------------"

echo "total batches: "$total_batch_number
echo "* use the batch count in the config of the training for the total batch count, when using these split files"
echo "total lines:   "$total_split_lines
