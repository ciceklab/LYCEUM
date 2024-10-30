# indexing bam file
for filename in ./finetune_example_data/bams/*.bam; do
        echo $filename
        samtools index $filename 

done

# create a folder for read depth data of the samples
mkdir finetune_read_depths

# generating the read depth data of the samples
for filename in ./finetune_example_data/bams/*.bam; do
    basename $filename
    f="$(basename -- $filename)"
    sambamba depth base -L hg38_hglft_genome_64dc_dcbaa0_unique.bed $filename > ./finetune_read_depths/"$f.txt"

done

mkdir processed_finetuning_samples

# run the preprocess script preprocess_sample.py
python ./scripts/finetune_preprocess_sample.py --readdepth ./finetune_read_depths --output ./processed_finetuning_samples --groundtruth ./finetune_example_data/ground_truth_labels --target hg38_hglft_genome_64dc_dcbaa0_unique.bed

mkdir processed_exonWise_finetuning_dataset

python ./scripts/finetune_create_dataset.py --input ./processed_finetuning_samples/ --output ./processed_exonWise_finetuning_dataset

# generating mean/std lookup file for ancient samples
python ./scripts/mean_std_calculator.py --processed_samples ./processed_finetuning_samples --setname toySet_ft --output ./stats/