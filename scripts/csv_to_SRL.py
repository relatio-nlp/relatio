import argparse
import ast
from pathlib import Path
from typing import List

from ipyparallel.error import UnmetDependency
from tqdm import tqdm

CODE_PATH = Path(
    "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/narrative-nlp/code"
).resolve()

SRL_MODEL_PATH = Path(
    "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/Andrei/SRL/srl_model/bert-base-srl-2020.03.24.tar.gz"
).resolve()
DOCUMENTS_PATH = Path(
    "/cluster/work/lawecon/Projects/Ash_Gauthier_Widmer/gpo_for_srl/"
).resolve()

SRL_OUTPUT_PATH = DOCUMENTS_PATH.with_name(DOCUMENTS_PATH.name + "_output")

MAX_CHAR_LENGTH_ON_CORE = 2_000
MAX_BATCH_CHAR_LENGTH = 20_000

# isort:skip
import sys  # isort:skip

sys.path.append(str(CODE_PATH))  # isort:skip
from ipcluster import LeonhardIPCluster  # isort:skip
from semantic_role_labeling import estimate_time, output_file  # isort:skip


def char_length(filepath: Path, correction: int):
    return len(filepath.read_text()) - correction


def batch_filepaths(filepaths: List[Path], max_time: int = 4 * 60 * 60):
    batches: List[List[Path]] = []
    batch: List[Path] = []
    time = 0
    for filepath in filepaths:
        new_time = estimate_time(char_length(filepath, correction=len("sentence\n")))
        if time + new_time > max_time:
            batches.append(batch)
            batch = [filepath]
            time = new_time
        else:
            batch.append(filepath)
            time += new_time
    if batch:
        batches.append(batch)

    return batches


def split_in_batches(glob_pattern) -> Path:
    filepaths = list(sorted(DOCUMENTS_PATH.glob(glob_pattern)))
    filepaths = [
        filepath
        for filepath in filepaths
        if not output_file(filepath, SRL_OUTPUT_PATH).exists()
    ]
    batches = batch_filepaths(filepaths)
    for i in range(100):
        if not (SRL_OUTPUT_PATH / ("series" + str(i).zfill(2))).exists():
            break

    batch_output_path = SRL_OUTPUT_PATH / ("series" + str(i).zfill(2))
    batch_output_path.mkdir()

    for batch_index, batch in enumerate(batches, start=1):
        (batch_output_path / (str(batch_index) + ".txt")).write_text(
            str([str(el) for el in batch])
        )
    return batch_output_path


def run_from_batch(batch_path: Path):
    ipcluster = LeonhardIPCluster()
    ipcluster.start()
    ipcluster.connect()

    cuda_device_ids = ipcluster.assign_cuda_device()
    rc = ipcluster.rc

    dview = rc[:]
    dview.block = True
    dview.scatter("cuda_device", cuda_device_ids)

    srl_kwargs = {
        "path": str(SRL_MODEL_PATH),
        "max_batch_char_length": MAX_BATCH_CHAR_LENGTH,
    }

    dview.push(
        {
            "CODE_PATH": str(CODE_PATH),
            "SRL_KWARGS": srl_kwargs,
            "SRL_OUTPUT_PATH": SRL_OUTPUT_PATH,
        }
    )

    dview.execute(
        """
    import ast
    import json
    import sys

    from ipyparallel.error import UnmetDependency

    sys.path.append(CODE_PATH)
    from semantic_role_labeling import SRL, output_file

    CUDA_DEVICE = cuda_device[0]

    srl = SRL(cuda_device=CUDA_DEVICE, **SRL_KWARGS)
    """
    )

    lview = rc.load_balanced_view()
    lview.block = True

    @lview.parallel()
    def open_srl_save(filepath, max_char_length_on_core=MAX_CHAR_LENGTH_ON_CORE):
        with open(filepath, "r") as f:
            sentences = f.readlines()[1:]
        sentences_char_length = sum([len(" ".join(el.split())) for el in sentences])
        if sentences_char_length > max_char_length_on_core and CUDA_DEVICE == -1:
            raise UnmetDependency
        res = srl(sentences)

        with open(output_file(filepath, SRL_OUTPUT_PATH), "w") as f:
            json.dump(res, f)

    with open(batch_path, "r") as f:
        filepaths = ast.literal_eval(f.readline())
        filepaths = [Path(filepath) for filepath in filepaths]
    print(f"Starting SRL on {len(filepaths)} files")
    open_srl_save.map((filepath for filepath in filepaths))

    ipcluster.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        help="how to run the script",
        choices=["split", "run"],
        required=True,
    )
    parser.add_argument(
        "-i",
        "--batch_path",
        help="input batch path if the mode choice is run",
        default="",
    )
    parser.add_argument(
        "-g",
        "--glob_pattern",
        help="input glob_pattern used for matching files if the mode choice is split",
        default="",
    )

    args = parser.parse_args()

    SRL_OUTPUT_PATH.mkdir(exist_ok=True)

    if args.mode == "split":
        if args.glob_pattern == "":
            raise ValueError()
        res = split_in_batches(args.glob_pattern)
        print(str(res))
    elif args.mode == "run":
        batch_path = Path(args.batch_path)
        if args.batch_path == "" or batch_path.exists() is False:
            raise ValueError()
        run_from_batch(batch_path)
