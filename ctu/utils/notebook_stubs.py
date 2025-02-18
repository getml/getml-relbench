import base64
import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from unittest.mock import patch

import cairosvg
import getml
import jinja2
import jupytext
import requests
from db_transformer.data import CTUDataset
from db_transformer.data.dataset_defaults import (
    CTU_REPOSITORY_DEFAULTS,
)
from db_transformer.db.schema_autodetect import SchemaAnalyzer
from lalia.chat.messages import UserMessage
from lalia.chat.session import Session
from lalia.llm import OpenAIChat
from lalia.llm.models import ChatModel
from markdownify import markdownify as md
from nbconvert.preprocessors import ExecutePreprocessor
from pydantic.alias_generators import to_snake

from ctu.utils.data import (
    infer_task_type,
    load_ctu_dataset,
)

UTILS_ROOT = Path(__file__).parent
OUTPUT_PATH = UTILS_ROOT / "../stubs"


"""
Datasets used in the "Transformers Meet Relational Databases" paper.
"""
PAPER_DATASETS = [
    "Dallas",
    "financial",
    "mutagenesis",
    "Pima",
    "WebKP",
    "Same_gen",
    "PubMed_Diabetes",
    "Accidents",
    "imdb_ijs",
    "tpcd",
    "Basketball_men",
    "restbase",
    # "AdventureWorks2014", # conversion to torch.frame fails upstream
    "FNHK",
    "sakila",
    "stats",
    "Grants",
    "ConsumerExpenditures",
    "employee",
    "SalesDB",
    "Seznam",
]


def retrieve_dataset_description(
    dataset: Annotated[str, "The name of the dataset to retrieve information about."],
) -> str:
    """
    Retrieve the description of a dataset from the CTU dataset repository.
    """
    dataset_url = dataset.replace("_", "").replace("-", "")
    url = f"https://relational.fel.cvut.cz/dataset/{dataset_url}"
    response = requests.get(url)
    response.raise_for_status()
    return md(response.text)


PROMPT = """
Write a small dataset description, follow the structure below:
- the data model (relational schaema)
- the task (classification, regression) and target column
- the types of the columns
- some metadata about the dataset (size, number of tables, number of rows, etc.)

As all datasets are well known public datasets extensively used in research, augmentthe description with
additional information known about those datasets, i.e. which research papers used them, what kind of tasks
they are used for in general, etc.

AVOID BOLD TEXT AND PREFER ITALICS FOR EMPHASIS.

THE TEXT WILL BE EMBEDDED IN A NOTEBOOK STUB FOR THE DATASET. SO PLEASE DO NOT GENERATE ANY
TOP-LEVEL MARKUP (HEADLINES, ETC.). JUST THE TEXT CONTENT. THANK YOU!
"""

llm = OpenAIChat(
    api_key=os.environ["OPENAI_API_KEY"], model=ChatModel.GPT_4O, temperature=0
)
session = Session(
    llm=llm,
    system_message=PROMPT,
    functions=[retrieve_dataset_description],
)


def create_notebook_stub(dataset: str, output_path: Path = OUTPUT_PATH):
    with open(UTILS_ROOT / "assets/notebook_template.jinja2") as f:
        template = jinja2.Template(f.read())

    dataset_defaults = CTU_REPOSITORY_DEFAULTS[dataset]

    dataset_er_diagram_url = (
        f"https://relational.fel.cvut.cz/assets/img/datasets-generated/{dataset}.svg"
    )
    diagram_svg = requests.get(dataset_er_diagram_url).content
    diagram_png = cairosvg.svg2png(bytestring=diagram_svg)

    diagram_png_base64 = base64.b64encode(diagram_png).decode("utf-8")

    diagram_message = UserMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{diagram_png_base64}"},
            },
        ]
    )
    session.messages.add(diagram_message)

    population, peripheral = load_ctu_dataset(dataset, as_pandas=True)
    population_name = to_snake(dataset_defaults.target_table).replace(" ", "_")
    peripheral_names = [to_snake(name).lower() for name in sorted(peripheral)]

    task_type = infer_task_type(dataset_defaults, population)

    dataset_description = session(
        f"Use the datamodel above to create a dataset description for the '{dataset}' dataset, "
        "first retrieve additonal information through the `retrieve_dataset_description` function. "
        "Particularly, describe the target column and the task (classification, regression) of the dataset. "
        f"The task type for this dataset is: {task_type!r}."
    ).content

    py_percent_rendered = template.render(
        project_name=f"{to_snake(dataset)}",
        dataset=dataset,
        dataset_description=dataset_description,
        dataset_er_diagram_url=dataset_er_diagram_url,
        target_column=dataset_defaults.target_column,
        task_type=task_type,
        population_name=population_name,
        peripheral_names=peripheral_names,
    )

    notebook = jupytext.reads(py_percent_rendered, fmt="py:percent")

    # execute the notebook to populate output cells
    exec_proc = ExecutePreprocessor(timeout=None)
    exec_proc.preprocess(notebook)

    jupytext.write(notebook, output_path / f"{to_snake(dataset)}.ipynb")


def create_notebook_stubs():
    output_path = OUTPUT_PATH
    for dataset in PAPER_DATASETS:
        print(f"Creating notebook stub for {dataset}... ")
        if Path(output_path / f"{to_snake(dataset)}.ipynb").exists():
            print("Already exists. ")
        else:
            create_notebook_stub(dataset, output_path)
            print("Done.")
