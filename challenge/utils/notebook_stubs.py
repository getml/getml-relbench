import base64
import os
from pathlib import Path
from typing import Annotated
from unittest.mock import patch

import cairosvg
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
from lalia.llm.models import ChatModel
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
    "PubMed_Diabetes",
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
Write a small dataset description, follow the structure below:
- the data model (relational schaema)
- the task (classification, regression) and target column
- the types of the columns
- some metadata about the dataset (size, number of tables, number of rows, etc.)

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
        lambda: getml.utilities.progress._is_jupyter()
        and not getml.utilities.progress._is_emacs_kernel(),
    ):
        schema = analyzer.guess_schema()

    population_name = to_snake(CTU_REPOSITORY_DEFAULTS[dataset].target_table)
    peripheral_names = [
        to_snake(tale_name)
        for tale_name in schema
        if tale_name != CTU_REPOSITORY_DEFAULTS[dataset].target_table
    ]

    dataset_er_diagram_url = (
        f"https://relational.fel.cvut.cz/assets/img/datasets-generated/{dataset}.svg"
    )
    svg_data = requests.get(dataset_er_diagram_url).content
    png_data = cairosvg.svg2png(bytestring=svg_data)

    base64_image = base64.b64encode(png_data).decode("utf-8")

    message = UserMessage(
        content=[
            {
                "type": "text",
                "text": f"Use the datamodel below to create a dataset description for the '{dataset}' dataset, "
                "first retrieve additonal information through the `retrieve_dataset_description` function.",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            },
        ]
    )
    session.messages.add(message)

    dataset_description = session("").content

    py_percent_rendered = template.render(
        project_name=f"db_transformer_{to_snake(dataset)}",
        dataset=dataset,
        dataset_description=dataset_description,
        dataset_er_diagram_url=dataset_er_diagram_url,
        target_column=CTU_REPOSITORY_DEFAULTS[dataset].target_column,
        population_name=population_name,
        peripheral_names=peripheral_names,
    )

    notebook = jupytext.reads(py_percent_rendered, fmt="py:percent")

    # execute the notebook to populate output cells
    exec_proc = ExecutePreprocessor(timeout=None)
    exec_proc.preprocess(notebook)

    jupytext.write(notebook, output_path / f"{to_snake(dataset)}.ipynb")
    return session


def create_notebook_stubs():
    output_path = OUTPUT_PATH
    for dataset in PAPER_DATASETS:
        print(f"Creating notebook stub for {dataset}... ")
        if Path(output_path / f"{to_snake(dataset)}.ipynb").exists():
            print("Already exists. ")
        else:
            create_notebook_stub(dataset, output_path)
            print("Done.")
