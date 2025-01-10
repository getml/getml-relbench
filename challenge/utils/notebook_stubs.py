import os
from pathlib import Path
from typing import Annotated
from unittest.mock import patch

import getml
import jinja2
import jupytext
import requests
from db_transformer.data import CTUDataset
from db_transformer.data.dataset_defaults import CTU_REPOSITORY_DEFAULTS
from db_transformer.db.schema_autodetect import SchemaAnalyzer
from lalia.chat.messages import UserMessage
from lalia.chat.session import Session
from lalia.llm import OpenAIChat
from markdownify import markdownify as md
from nbconvert.preprocessors import ExecutePreprocessor
from pydantic.alias_generators import to_snake

UTILS_ROOT = Path(__file__).parent
OUTPUT_PATH = UTILS_ROOT / "../tasks"


"""
Datasets used in the "Transformers Meet Relational Databases" paper.
"""
PAPER_DATASETS = [
    "Carcinogenesis",
    "CraftBeer",
    "Dallas",
    "financial",
    "Mondial",
    "MuskSmall",
    "mutagenesis",
    "Pima",
    "PremierLeague",
    "Toxicology",
    "UW_std",
    "WebKP",
    "DCG",
    "Same_gen",
    "voc",
    "PubMed",
    "Accidents",
    "imdb_ijs",
    "tpcd",
]


def retrieve_dataset_description(
    dataset: Annotated[str, "The name of the dataset to retrieve information about."],
) -> str:
    """
    Retrieve the description of a dataset from the CTU dataset repository.
    """
    url = f"https://relational.fel.cvut.cz/dataset/{dataset}"
    response = requests.get(url)
    response.raise_for_status()
    return md(response.text)


PROMPT = """
Write a small dataset description for me. The description should include:
- the data model (relational schaema)
- the types of the columns
- some metadata about the dataset (size, number of tables, number of rows, etc.)

THE TEXT WILL BE EMBEDDED IN A NOTEBOOK STUB FOR THE DATASET. SO PLEASE DO NOT GENERATE ANY
TOP-LEVEL MARKUP (HEADLINES, ETC.). JUST THE TEXT CONTENT. THANK YOU!
"""

llm = OpenAIChat(api_key=os.environ["OPENAI_API_KEY"])
session = Session(
    llm=llm,
    system_message=PROMPT,
    functions=[retrieve_dataset_description],
)


def create_notebook_stub(dataset: str, output_path: Path = OUTPUT_PATH):
    with open(UTILS_ROOT / "assets/notebook_template.jinja2") as f:
        template = jinja2.Template(f.read())

    conn = CTUDataset.create_remote_connection(dataset)
    analyzer = SchemaAnalyzer(
        conn,
        verbose=True,
        target=CTU_REPOSITORY_DEFAULTS[dataset].target,
        target_type=CTU_REPOSITORY_DEFAULTS[dataset].task.to_type(),
        post_guess_schema_hook=CTU_REPOSITORY_DEFAULTS[dataset].schema_fixer,
    )

    with patch(
        "db_transformer.helpers.progress.is_notebook",
        getml.utilities.progress._is_jupyter_without_ipywidgets,
    ):
        schema = analyzer.guess_schema()

    population_table_name = CTU_REPOSITORY_DEFAULTS[dataset].target_table
    peripheral_table_names = [
        tale_name for tale_name in schema if tale_name != population_table_name
    ]

    dataset_er_diagram_url = (
        f"https://relational.fel.cvut.cz/assets/img/datasets-generated/{dataset}.svg"
    )
    message = UserMessage(
        content=[
            {
                "type": "text",
                "text": "Please provide a dataset description of the dataset in the image.",
            },
            {"type": "image", "url": dataset_er_diagram_url},
        ]
    )

    dataset_description = session(dataset).content

    py_percent_rendered = template.render(
        dataset=dataset,
        dataset_description=dataset_description,
        dataset_er_diagram_url=dataset_er_diagram_url,
        population_table_name=population_table_name,
        peripheral_table_names=peripheral_table_names,
    )

    notebook = jupytext.reads(py_percent_rendered, fmt="py:percent")

    # execute the notebook to populate output cells
    exec_proc = ExecutePreprocessor(timeout=None)
    exec_proc.preprocess(notebook)

    jupytext.write(notebook, output_path / f"{to_snake(dataset)}.ipynb")


def create_notebook_stubs():
    for dataset in PAPER_DATASETS:
        print(f"Creating notebook stub for {dataset}... ", end="")
        create_notebook_stub(dataset)
        print("Done.")
