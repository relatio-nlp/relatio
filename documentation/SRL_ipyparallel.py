from pathlib import Path
import time

import ipyparallel as ipp

rc = ipp.Client()


t0 = time.time()

ONLY_CPU = False

dview = rc[:]
dview.block = True
lview = rc.load_balanced_view()
lview.block = True


dview.execute(
    """
import os
import torch
total_cuda_devices = torch.cuda.device_count()
pid = os.getpid()
hostname = os.uname()[1]
"""
)


res = dview.pull(("total_cuda_devices", "hostname"))


cuda_device_ids = []
total_cuda_devices = {}
for el in res:
    if not el[1] in total_cuda_devices:
        total_cuda_devices[el[1]] = el[0]
    if total_cuda_devices[el[1]] > 0:
        total_cuda_devices[el[1]] -= 1
        cuda_device_id = total_cuda_devices[el[1]]
    else:
        cuda_device_id = -1
    cuda_device_ids.append(cuda_device_id)


if ONLY_CPU:
    cuda_device_ids = [-1] * len(cuda_device_ids)

print(res, cuda_device_ids)

dview.scatter("cuda_device", cuda_device_ids)


code_path = Path(
    "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/narrative-nlp/code"
).resolve()

srl_model_path = Path(
    "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/SRL/srl_model/srl-model-2018.05.25.tar.gz"
).resolve()

srl_model_path = Path(
    "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/SRL/srl_model/bert-base-srl-2020.03.24.tar.gz"
).resolve()


documents_path = Path(
    "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/cong_gpo_taxation_sentences/"
).resolve()
srl_output_path_prefix = documents_path / "srl_output"

for i in range(100):
    if not (srl_output_path_prefix / str(i).zfill(2)).exists():
        break
srl_output_path = srl_output_path_prefix / str(i).zfill(2)
srl_output_path.mkdir()


srl_kwargs = {
    "path": str(srl_model_path),
    "max_batch_char_length": 20_000,
    "max_sentence_length": 349,
}


dview.push(
    {
        "code_path": str(code_path),
        "srl_kwargs": srl_kwargs,
        "srl_output_path": srl_output_path,
    }
)


dview.execute(
    """
import ast
import json
import sys

sys.path.append(code_path)
from semantic_role_labeling import SRL

srl = SRL(cuda_device=cuda_device[0], **srl_kwargs)
"""
)

filenames = sorted(documents_path.glob("*.txt"))
sorted_filenames = sorted(filenames, key=lambda s: -s.stat().st_size)


def log_time(path, txt, t0):
    with open(path, "a") as f:
        dt = time.time() - t0
        f.write(f"{txt},{dt} [s]\n")


@lview.parallel()
def open_srl_save(filename):
    with open(filename, "r") as f:
        senteces = ast.literal_eval(f.readline())
    res = srl(senteces)

    with open((srl_output_path / filename.name).with_suffix(".json"), "w") as f:
        json.dump(res, f)


log_time(srl_output_path / "time.txt", "init", t0)

open_srl_save.map((filename for filename in sorted_filenames))

log_time(srl_output_path / "time.txt", "finish", t0)

rc.shutdown(hub=True)
