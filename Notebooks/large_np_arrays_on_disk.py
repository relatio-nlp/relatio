# %%
import math
from pathlib import Path
import pickle
import sys
from typing import Union

import numpy as np


def h_bytes(size: Union[int, float]) -> str:
    "Human readable bytes"
    log2 = math.log2(size)
    if log2 < 7:
        return f"{size} B"
    elif log2 < 17:
        return f"{size/(2**10):.2f} KiB"
    elif log2 < 27:
        return f"{size/(2**20):.2f} MiB"
    else:
        return f"{size/(2**30):.2f} GiB"


def size_bytes(arr, h=True):
    res = arr.size * arr.itemsize
    if h is True:
        return h_bytes(res)
    else:
        return res


# %%
a = np.random.rand(10 ** 9)
size_bytes(a)
# %%
h_bytes(sys.getsizeof(a))
# %%
file_path = Path("_to_delete_a.npy")
np.save(file_path, a)
h_bytes(file_path.stat().st_size)

# %%
file_path = Path("_to_delete_a.npz")
np.savez(file_path, a)
h_bytes(file_path.stat().st_size)


# %%
file_path = Path("_to_delete_a.pickle")
with open(file_path, "wb") as f:
    pickle.dump(a, f)
h_bytes(file_path.stat().st_size)
# %%
