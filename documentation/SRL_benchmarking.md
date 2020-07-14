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

**GeForceRTX2080Ti** 8xGPU nodes with ~ 10GB / GPU and 36 CORES with 10 GB / Core

bsub -n 1 -W 4:00 -R "rusage[mem=10000,ngpus_excl_p=1] select[gpu_model0==GeForceRTX2080Ti]" -o out.txt -e err.txt 

### Issues

Hard to predict memory requirements.