import argparse
import ast
from pathlib import Path
from typing import List, Tuple

from ipyparallel.error import UnmetDependency

CODE_PATH = Path(__file__).resolve().parent.parent / "code"

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


def split_in_batches(
    glob_pattern: str, DOCUMENTS_PATH: Path, SRL_OUTPUT_PATH: Path
) -> Tuple[Path, int]:

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
    return batch_output_path, len(batches)


def run_from_batch(batch_path: Path, SRL_MODEL_PATH: Path):
    SRL_OUTPUT_PATH = batch_path.parent.parent

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
        output_path = output_file(filepath, SRL_OUTPUT_PATH)
        if output_path.exists():
            pass
        else:
            with open(filepath, "r") as f:
                sentences = f.readlines()[1:]
            sentences_char_length = sum([len(" ".join(el.split())) for el in sentences])
            if sentences_char_length > max_char_length_on_core and CUDA_DEVICE == -1:
                raise UnmetDependency
            res = srl(sentences)

            with open(output_path, "w") as f:
                json.dump(res, f)

    with open(batch_path, "r") as f:
        filepaths = ast.literal_eval(f.readline())
        filepaths = [Path(filepath) for filepath in filepaths]
    print(f"Starting SRL on {len(filepaths)} files")
    open_srl_save.map((filepath for filepath in filepaths))

    ipcluster.stop()


def srl_output_path(DOCUMENTS_PATH: Path, suffix: str) -> Path:
    return DOCUMENTS_PATH.with_name(DOCUMENTS_PATH.name + suffix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        help="how to run the script",
        choices=["split", "run"],
        required=True,
    )
    group1 = parser.add_argument_group("split", "--mode split")
    group1.add_argument(
        "-d",
        "--documents_path",
        help="input documents folder path with csv",
        default="",
    )

    group2 = parser.add_argument_group("run", "--mode run")

    group2.add_argument(
        "-s", "--srl_model_path", help="SRL model path", default="",
    )
    group2.add_argument(
        "-b", "--batch_path", help="input batch path", default="",
    )
    group2.add_argument(
        "-g",
        "--glob_pattern",
        help="input glob_pattern used for matching files",
        default="",
    )

    args = parser.parse_args()

    if args.mode == "split":
        DOCUMENTS_PATH = Path(args.documents_path).resolve()
        if args.documents_path == "" or DOCUMENTS_PATH.exists() is False:
            raise ValueError(f"{DOCUMENTS_PATH} does not exists")

        if args.glob_pattern == "":
            raise ValueError()

        SRL_OUTPUT_PATH = srl_output_path(DOCUMENTS_PATH, suffix="_output")
        SRL_OUTPUT_PATH.mkdir(exist_ok=True)

        batch_path, len_batches = split_in_batches(
            args.glob_pattern, DOCUMENTS_PATH, SRL_OUTPUT_PATH
        )
        print(str(batch_path), "\n", len_batches)
    elif args.mode == "run":

        batch_path = Path(args.batch_path)
        SRL_MODEL_PATH = Path(args.srl_model_path).resolve()

        if args.batch_path == "" or batch_path.exists() is False:
            raise ValueError(f"{args.batch_path} does not exists")
        elif args.srl_model_path == "" or SRL_MODEL_PATH.exists() is False:
            raise ValueError(f"{args.srl_model_path} does not exists")

        run_from_batch(batch_path, SRL_MODEL_PATH)
