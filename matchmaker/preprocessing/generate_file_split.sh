# 
# bash script to split a file in n-similiar sized chunks (to be fed into the optimized multiprocess batch generators)
# also returns the correct batch count (that the multiprocess generators will produce)
#

# arg 1: base file
# arg 2: n chunks (should correspond to the number of processes you want to use)
# arg 3: output folder + prefix
# arg 4: batch size

if [ $# -ne 4 ]
  then
    echo "Usage (bash): ./generate_file_split.sh <base_file> <n file chunks> <output_folder> <batch_size>"
    exit 1
fi

base_file=$1
n_chunks=$2
output_folder_prefix=$3
batch_size=$4

# make sure the directory exists
output_folder=$(dirname "${output_folder_prefix}")
mkdir -p $output_folder

#
# script start
#

base_length=$(wc -l < $1)
echo "Base file line #: "$base_length
chunk_lines=$((base_length / n_chunks))
echo "Creating "$n_chunks" chunks, with ~"$chunk_lines" lines each"
echo "Output directory: "$output_folder
echo "--------------------------------------------------------------"

split --number=l/$n_chunks -d --additional-suffix ".tsv" $base_file $output_folder_prefix

total_batch_number=0
total_split_lines=0

for file in "$output_folder_prefix"*; do
    
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
