import os
import random
import tempfile
import warnings
from contextlib import redirect_stdout
from enum import StrEnum
from functools import lru_cache
from typing import Dict, Literal, Tuple, Union, overload
from unittest.mock import patch

import getml
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from db_transformer import data  # type: ignore # noqa: E402
from pydantic.alias_generators import to_snake

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


RELDB_IP = "35.195.45.191"


class RelDBDataset(data.CTUDataset):
    @classmethod
    def get_url(cls, dataset: data.CTUDatasetName) -> str:
        connector = "mysql+mysqlconnector"
        port = 3306
        if dataset == "tpcd":
            return f"{connector}://guest:relational@{RELDB_IP}:{port}/{dataset}"
        return super().get_url(dataset)


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"


def load_data_from_reldb_dataset(
    dataset: RelDBDataset,
    share_val: float = 0.3,
    share_test: float = 0.0,
    as_pandas: bool = False,
) -> Union[
    Tuple[getml.data.DataFrame, Dict[str, getml.data.DataFrame]],
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame]],
]:
    with patch(
        "db_transformer.helpers.progress.is_notebook",
        lambda: getml.utilities.progress._is_jupyter()
        and not getml.utilities.progress._is_emacs_kernel(),
    ):
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hetero_data, _ = dataset.build_hetero_data()

    population_name = dataset.defaults.target_table

    n_total = hetero_data[population_name].y.shape[0]
    splitted = T.RandomNodeSplit(
        "train_rest",
        num_val=int(share_val * n_total),
        num_test=(share_test * n_total),
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

    dfs = {
        name: table.df.drop("__filler", axis=1, errors="ignore")
        for name, table in dataset.db.table_dict.items()
    }

    population = dfs.pop(population_name).join(split)
    peripheral = dfs

    if as_pandas:
        return population, peripheral

    if (
        dataset.defaults.task is data.dataset_defaults.utils.TaskType.CLASSIFICATION
        and population[dataset.defaults.target_column].nunique() > 2
    ):
        target_role = getml.data.roles.unused_string
    else:
        target_role = getml.data.roles.target
    population_getml = getml.data.DataFrame.from_pandas(
        population,
        name=to_snake(population_name).replace(" ", "_"),
        roles={target_role: [dataset.defaults.target_column]},
    )

    return population_getml, {
        to_snake(name): getml.data.DataFrame.from_pandas(df, name=to_snake(name))
        for name, df in peripheral.items()
    }


def build_reldb_dataset(
    name: str, share_val: float = 0.3, share_test: float = 0.0, as_pandas: bool = False
) -> RelDBDataset:
    if not share_val:
        raise ValueError("share_val must be greater than 0")

    with patch(
        "db_transformer.helpers.progress.is_notebook",
        lambda: getml.utilities.progress._is_jupyter()
        and not getml.utilities.progress._is_emacs_kernel(),
    ):
        return RelDBDataset(
            name,
            data_dir=f"{tempfile.gettempdir()}/ctu_data",
            save_db=False,  # serialization is broken
        )


@overload
def load_ctu_dataset(
    name: str,
    share_val: float = 0.3,
    share_test: float = 0.0,
    as_pandas: Literal[False] = False,
) -> Tuple[getml.data.DataFrame, Dict[str, getml.data.DataFrame]]: ...


@overload
def load_ctu_dataset(
    name: str,
    share_val: float = 0.3,
    share_test: float = 0.0,
    as_pandas: Literal[True] = True,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]: ...


def load_ctu_dataset(
    name: str,
    share_val: float = 0.3,
    share_test: float = 0.0,
    as_pandas: bool = False,
) -> Union[
    Tuple[getml.data.DataFrame, Dict[str, getml.data.DataFrame]],
    Tuple[pd.DataFrame, Dict[str, pd.DataFrame]],
]:
    """
    Load a CTU dataset as a getML data frame and split it into train, validation, and test sets.

    The split reproduces the original split from the "Transformers meets Relational Databases" paper.
    """
    dataset = build_reldb_dataset(name, share_val, share_test, as_pandas)
    return load_data_from_reldb_dataset(dataset, share_val, share_test, as_pandas)


def infer_task_type(
    dataset_defaults: data.dataset_defaults.CTUDatasetDefault,
    population: pd.DataFrame,
) -> TaskType:
    if dataset_defaults.task is data.dataset_defaults.utils.TaskType.CLASSIFICATION:
        if population[dataset_defaults.target_column].nunique() > 2:
            return TaskType.MULTICLASS_CLASSIFICATION
        return TaskType.CLASSIFICATION
    return TaskType.REGRESSION
