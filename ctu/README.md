# Data Science Challenge ­ getML Community (CTU datasets)

In this part of the getML Data Science Challenge, you are working on data sets from [CTU Relational Dataset Repository](https://relational.fel.cvut.cz/).

## Getting Started
To ease working on the datasets, we have already generated stubs for all published tasks. You can find all published stubs [stubs](stubs) subfolder.
You can just work on the stubs, the data loading should work right out of the box. When starting to work on the tasks, the first two thing you should do are
probably [annotating the data](https://getml.com/latest/user_guide/concepts/annotating_data/) and [defining a data model](https://getml.com/latest/user_guide/concepts/data_model/).
When you start working on a task, please copy the respective stub to the `ctu` subfolder (this directory) of the repository and start working on it:

```
cp stubs/financial.ipynb .
```

#### Linux
To get started working on ctu notebooks on linux, just [install uv](https://docs.astral.sh/uv/getting-started/installation/) and leverage our [curated environment](pyproject.toml) with the `ctu` extra:
> [!NOTE]
> As other parts of the challenge require conflicting dependencies, we had to put conflicting dependencies in a separate dependency group (extra).
> Make sure to also install dependencies of the respective group, to do so, supply the `--extra` argument to the respective `uv` subcommand:
> `uv sync --extra ctu`

```sh
uv run --extra ctu jupyter lab
```

#### macOS and Windows
To get started on macOS and Windows, you first need to [start the getML docker service](https://getml.com/latest/install/packages/docker/):
```sh
curl https://raw.githubusercontent.com/getml/getml-community/1.5.0/runtime/docker-compose.yml | docker compose -f - up
```
Afterwards, [install uv](https://docs.astral.sh/uv/getting-started/installation/) and use the [provided environment](pyproject.toml) as above:
```sh
uv run --extra ctu jupyter lab
```

## Pick a Challenge
| **Dataset**                                                     | **Task**                                                                          | **PR's & Submissions**                                                                       | **Task + Measure**     | **Score getML**                  | **Score RDL** | **Score Human** |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------- | -------------------------------- | ------------- | --------------- |
| [financial](https://relational.fel.cvut.cz/dataset/Financial) | Load default       | [PR ???]()                                    | classification  | –                                |      |            |

