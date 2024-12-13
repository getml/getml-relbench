# %%

import getml
import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task
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
    categorical=["category"]
)

product_df = pd.read_parquet(f"{dataset.cache_dir}/db/product.parquet")
product_df.category = product_df.category.apply(lambda x:str(x))
product = getml.DataFrame.from_pandas(product_df,
name = 'product',
roles=product_roles)
# product = getml.DataFrame.from_parquet(
#     f"{dataset.cache_dir}/db/product.parquet",
#     name="product",
#     roles=product_roles,
# )
# %%
