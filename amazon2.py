# %%
import getml
import polars as pl
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from sentence_transformers import SentenceTransformer

# %%
dataset = get_dataset("rel-amazon", download=True)
task = get_task("rel-amazon", "item-churn", download=True)

getml.set_project("rel-amazon")

# %%

population_roles = getml.data.Roles(
    join_key=["product_id"],
    target=["churn"],
    time_stamp=["timestamp"],
)

subsets = ("train", "test", "val")
populations = {}
for subset in subsets:
    populations[subset] = getml.data.DataFrame.from_parquet(
        f"{task.cache_dir}/{subset}.parquet",
        name=f"population_{subset}",
        roles=population_roles,
    )
    populations[subset]["reference_date"] = dataset.test_timestamp.to_datetime64()
    populations[subset].set_role(["reference_date"], getml.data.roles.time_stamp)

abridged_pop = populations["train"][populations["train"].timestamp > getml.data.time.datetime(year=2015, month = 1, day=1)]
populations['train'] = abridged_pop
# %%
customer_roles = getml.data.Roles(
    join_key=["customer_id"]
)

customer = getml.DataFrame.from_parquet(
    f"{dataset.cache_dir}/db/customer.parquet",
    name="customer",
    roles=customer_roles,
)
# %%

product_roles = getml.data.Roles(
    join_key=["product_id"],
    numerical=["price"],
    categorical=["category", "brand"]
)

product_df = pl.scan_parquet(f"{dataset.cache_dir}/db/product.parquet")
product_df = product_df.with_columns(pl.col('category').list.sort(descending=True))
product_df = product_df.with_columns(
    pl.col("category").cast(pl.List(pl.String)).list.join(", ")
)

review_df_orig = pl.scan_parquet(f"{dataset.cache_dir}/db/review.parquet")
# %%
model = SentenceTransformer('snowflake/snowflake-arctic-embed-xs')
model.encode(review_df_orig.limit(10000).select('review_text').collect().to_series().to_list())