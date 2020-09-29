# Running job arrays for CSV to SRL on Leonhard

The documentation for job arrays is available at [https://scicomp.ethz.ch/wiki/Job_arrays](https://scicomp.ethz.ch/wiki/Job_arrays) .

We assume that 4h queue is desired (the fastest queue).

## Script `csv_to_SRL.py`

It is in the repo `narrative-nlp/scripts/csv_to_SRL.py` . The csv are assumed to have one line header and each sentence is on a separate line.

A script can split and run the batches.

Please check the help

```bash
python /cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/narrative-nlp/scripts/csv_to_SRL.py --help
```


## Split - making the batches for job arrays

This step can be done interactively. For 1 million sentences it should not take more than 30 min. 

Ask for an interactive job:
```bash
$ bsub -n 1 -Is -W 1:00 -R "rusage[mem=10000]" bash
```


Load the desired modules and activate the conda environment

```bash
$ module load eth_proxy openmpi/4.0.1 cuda/10.0.130
$ conda activate narrative-nlp-2
```

Run the script to split in batches. :

```bash
$ python "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/narrative-nlp/scripts/csv_to_SRL.py" \
--mode split \
--documents_path "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/gpo_for_srl" \
--glob_pattern "*-10-*_*.csv"
```


The output contains:
- on the first line the path with the batches, 
- and on the second line the number of batches, called next `len_batches` 

You need them to launch next the job arrays.

## Running the job arrays

Use a small bash file `run_job_array_csv_to_SRL.sh` with a similar content:

```bash
source ~/.bashrc
module load eth_proxy openmpi/4.0.1 cuda/10.0.130

conda activate narrative-nlp-2

python "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/narrative-nlp/scripts/csv_to_SRL.py" \
--mode run \
--srl_model_path "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/SRL/srl_model/bert-base-srl-2020.03.24.tar.gz" \
--batch_path "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/gpo_for_srl_output/series00/${LSB_JOBINDEX}.txt" 
```

**You might need to update the arguments**.

To launch the job array we assume at most 16 jobs running in parallel (keep in mind that each job will use 1 GPU - so this corresponds to up to 16 GPUs in parallel). 

```bash
$ bsub -G ls_lawecon -J "gpo_srl_OCT[1-len_batches]%16" -n 4 -W 4:00 -R "rusage[mem=10000,ngpus_excl_p=1] select[gpu_model0==GeForceRTX2080Ti]" < run_job_array_csv_to_SRL.sh
```

You can easily monitor the progress checking the job arrays (see the corresponding documentation) or even checking the number of files in the desired output folder, e.g.

```bash
$ ls -l /cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/gpo_for_srl_output/ | wc -l
```