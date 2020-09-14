# Running job arrays for CSV to SRL on Leonhard

## Preparation - making the batches for job arrays

The some parameters are hardcoded in csv_to_SRL.py file. 

We assume that 4h jobs on a GPU is used.

```bash
$ python  /cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/narrative-nlp/scripts/csv_to_SRL.py -m split -g "*-10-*_*.csv"

```

The output is precisely the path that we will need to run the job arrays. Use it in the script `run_job_array_csv_to_SRL.sh`.

Check how many files are there, e.g. 12, called number_of_files . You will need it in the next step.

## Running the job arrays

```bash
$ bsub -J "gpo_srl_OCT[1-number_of_files]" -n 4 -W 4:00 -R "rusage[mem=10000,ngpus_excl_p=1] select[gpu_model0==GeForceRTX2080Ti]" < run_job_array_csv_to_SRL.sh
```
