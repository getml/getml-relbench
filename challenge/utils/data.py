import os
import random
import tempfile
import warnings
from contextlib import redirect_stdout
from typing import Dict, Tuple
from unittest.mock import patch

import getml
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from db_transformer import data  # type: ignore # noqa: E402

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def load_ctu_dataset(
    name: str, share_val: float = 0.3, share_test: float = 0.0
) -> Tuple[getml.data.DataFrame, Dict[str, getml.data.DataFrame]]:
    """
    Load a CTU dataset as a getML data frame and split it into train, validation, and test sets.

    The split reproduces the original split from the "Transformers meets Relational Databases" paper.
    """
    if not share_val:
        raise ValueError("share_val must be greater than 0")

    with patch(
        "db_transformer.helpers.progress.is_notebook",
        getml.utilities.progress._is_jupyter_without_ipywidgets,
    ):
        dataset = data.CTUDataset(
            name,
            data_dir=f"{tempfile.gettempdir()}/ctu_data",
            force_remake=True,
            save_db=False,  # serialization is broken
        )
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hetero_data, _ = dataset.build_hetero_data()

    population_name = dataset.defaults.target_table

    n_total = hetero_data[population_name].y.shape[0]
    splitted = T.RandomNodeSplit(
        "train_rest", num_val=int(share_val * n_total), num_test=(share_test * n_total)
    )(hetero_data)
    population_hetero_data = splitted[population_name]

    train_mask, train_indices = population_hetero_data.train_mask.sort(descending=True)
    train_indices = train_indices[train_mask]

    val_mask, val_indices = population_hetero_data.val_mask.sort(descending=True)
    val_indices = val_indices[val_mask]

    test_mask, test_indices = population_hetero_data.test_mask.sort(descending=True)
    test_indices = test_indices[test_mask]

    index = torch.cat([train_indices, val_indices, test_indices], dim=0).numpy()
    subset = np.array(["train", "val", "test"]).repeat(
        [len(train_indices), len(val_indices), len(test_indices)]
    )

    split = pd.DataFrame({"split": subset, "index": index}).set_index("index")

    dfs = {name: table.df for name, table in dataset.db.table_dict.items()}

    population = dfs.pop(population_name).join(split)
    peripheral = dfs

    return getml.data.DataFrame.from_pandas(population, name=population_name), {
        name: getml.data.DataFrame.from_pandas(df, name=name)
        for name, df in peripheral.items()
    }
