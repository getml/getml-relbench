# getML Community x Relbench

## Getting Started
### Linux
To get started on linux, just [install uv](https://docs.astral.sh/uv/getting-started/installation/) and leverage our [curated environment](pyproject.toml):

```sh
uv run jupyter lab
```
### macOS and Windows
To get started on macOS and Windows, you first need to [start the getML docker service](https://getml.com/latest/install/packages/docker/):

```sh
curl https://raw.githubusercontent.com/getml/getml-community/1.5.0/runtime/docker-compose.yml | docker-compose up -f -
```

Afterwards, [install uv](https://docs.astral.sh/uv/getting-started/installation/) and use the [provided environment](pyproject.toml) as above:

```sh
uv run jupyter lab
```

