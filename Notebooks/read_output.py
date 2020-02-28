# %%
import pandas as pd
import json

import sys

sys.path.append("../code")

from utils import UsedRoles
from cooccurrence import CoOccurence

# %%
# Read df, labels and previously used roles
df = pd.read_pickle("./df.pkl")
with open("labels.json", "r") as f:
    labels = json.load(f)
    for role, value in labels.items():
        labels[role] = {int(k): v for k, v in value.items()}
with open("used_roles.json", "r") as f:
    used_roles = UsedRoles(json.load(f))

# %%
# Run cooccurence

cooc = CoOccurence(df, labels, used_roles)
cooc.subset = {"ARGO", "ARG1", "B-V"}
cooc.narratives_counts
cooc.narratives_pmi

