# SRL Benchmarking

## Test details

### Input data

The documents used as test are located on Leonhard at `/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/cong_gpo_taxation_sentences/*.txt` . Each document can be easily read

```python
import ast
from pathlib import Path

parent_path = Path(
    "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/cong_gpo_taxation_sentences/"
).resolve()

filepaths = sorted(parent_path.glob("*.txt"))

for filepath in filepaths:
    with open(filepath, "r") as f:
        sentences = ast.literal_eval(f.readline())
```

### Architecture

**GeForceRTX2080Ti** 8xGPU nodes with ~ 10.5GB / GPU and 36 CORES with 10.5GB / Core

```bash

bsub -n 1 -W 4:00 -R "rusage[mem=10500,ngpus_excl_p=1] select[gpu_model0==GeForceRTX2080Ti]"

```

A full node can be guaranteed using `-R fullnode`  

```bash

bsub -n 36 -W 4:00 -R fullnode -R "select[gpu_model0==GeForceRTX2080Ti]" 

```

### Results

| Implementation   | batch size \[sen or char\] | max sentence len \[char\] | COREs | GPUs | SRL execution time \[s\] | Run time \[s\] | Time for cores at the end \[s\] |
| ---------------- | -------------------------- | ------------------------- | ----- | ---- | ------------------------ | -------------- | ------------------------------- |
| Ref (Joblib)     | 80 sen                     | 349                       | 18    | 0    | 2361                     | 2402           |                                 |
| GPU              | 80 sen                     | 349                       | 0     | 1    | 1805                     | 1852           |                                 |
| GPU              | 20'000 char                | 349                       | 0     | 1    | 1619                     | 1663           |                                 |
| IPyParallel      | 20'000 char                | 349                       | 18    | 0    | 1838                     | 1997           | 160                             |
| IPyParallel      | 20'000 char                | 349                       | 0     | 1    | 1649                     | 1787           |                                 |
| IPyParallel      | 20'000 char                | 349                       | 0     | 2    | 764                      | 875            |                                 |
| IPyParallel      | 20'000 char                | 349                       | 0     | 4    | 378                      | 511            |                                 |
| IPyParallel      | 20'000 char                | 349                       | 14    | 4    | 1171                     | 1337           | 700                             |
| IPyParallel+BERT | 20'000 char                | 349                       | 0     | 1    | 1234                     | 1362           |                                 |
| IPyParallel+BERT | 80'000 char                | 349                       | 0     | 1    | 1140                     | 1268           |                                 |
| IPyParallel+BERT | 20'000 char                | 349                       | 18    | 0    | 1352                     | 1480           |                                 |
| IPyParallel+BERT | 80'000 char                | 349                       | 18    | 0    | 1747                     | 1935           | 400                             |
| IPyParallel+BERT | 20'000 char                | 349                       | 1     | 0    | 15375                    | 15497          |                                 |                       |

When *n* GPUs were used there also *n* cores required but not mentioned in the table above because they were not used to perform SRL but only for the communication with the GPU.

Even if no GPU is required by the code one can require one in the`bsub` command to increase job priority (this was done in practice).

The implementation for IPyParallel is available [here](SRL_ipyparallel.py) and the tests were executed from `/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/SRL
`

#### Executing jobs

```bash
bsub -n 18 -W 2:00 -R "rusage[mem=10500,ngpus_excl_p=1] select[gpu_model0==GeForceRTX2080Ti] span[ptile=18]" -J "SRL" 'source ~/.bashrc && module load eth_proxy openmpi/4.0.1 cuda/10.0.130 && conda activate narrative-nlp && python benchmarking_srl.py'

bsub -n 18 -W 2:00 -R "rusage[mem=10500,ngpus_excl_p=1] select[gpu_model0==GeForceRTX2080Ti] span[ptile=18]" -J "SRL" 'source ~/.bashrc && module load eth_proxy openmpi/4.0.1 cuda/10.0.130 && conda activate narrative-nlp && ipcluster start --ip="*" --location=$(hostname) --engines=MPI -n $LSB_MAX_NUM_PROCESSORS --daemonize=True && sleep 60 && python SRL_ipyparallel.py'
```

### Observations

- GPU integration and parallezation via IPython Parallel was done successfully
- the memory requirements are hard to anticipate - we might have the surprise that a running job is killed because it requires more than allocated
- one can just use 1 GPU without the need for IPython Parallel 
- execution time for 1 GPU is similar with Reference results for 18 Cores as done previously using Joblib
- execution time with 4 GPUs is as expected - 1/4 of the previous execution time
- for small jobs it is discouraged to combine cores and GPUs because the execution time one one core can be very long and therefore it might keep the other GPUs / Cores busy but actually doing nothing
- I realized that running multiple jobs using ipython parallel at the same time can be tricky (each run should use a different ipython parallel cluster)
- IPython Parallel shines for interactive work or doing just one big job 
- for SRL one might assess job arrays https://scicomp.ethz.ch/wiki/Job_arrays , where each job can use 1 GPU