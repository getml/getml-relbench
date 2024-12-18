# %%

import getml
import pandas as pd
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

product_df = pd.read_parquet(f"{dataset.cache_dir}/db/product.parquet")
product_df.category = product_df.category.apply(lambda x:str(x))
product = getml.DataFrame.from_pandas(product_df,
name = 'product',
roles=product_roles)

# %%
review_df = pd.read_parquet(f"{dataset.cache_dir}/db/review.parquet")
# %%

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings
#review_df = review_df.head(100)
review_df['embedding'] = model.encode(review_df['review_text']).tolist()
embedding_columns = columns=[f"dim_{i}" for i in range(len(review_df['embedding'][0]))]
embedding_df = pd.DataFrame(review_df['embedding'].tolist(), columns=embedding_columns)
review_df.drop('embedding', inplace=True, axis=1)

review_df = pd.concat([review_df, embedding_df], axis=1)

review_roles = getml.data.Roles(
    time_stamp=["review_time"],
    join_key=["customer_id", "product_id"],
    numerical=embedding_columns + ["rating"],
    categorical=['verified']
)
# %%
review = getml.DataFrame.from_pandas(review_df,
name = 'review',
roles=review_roles)

print(review)

# %%


dm = getml.data.DataModel(population=populations["train"].to_placeholder())

dm.add(
    customer.to_placeholder(), product.to_placeholder(), review.to_placeholder()
)
dm.population.join(
    dm.product, on="product_id", relationship=getml.data.relationship.many_to_one
)
dm.population.join(
    dm.review, on="product_id", time_stamps=("timestamp", "review_time")
)
# dm.review.join(
#     dm.customer, on="customer_id", relationship=getml.data.relationship.many_to_one
# )
# %%
container = getml.data.Container(**populations)
container.add(customer, product, review)
# %%
pipe = getml.Pipeline(
    tags=["task: item-churn"],
    data_model=dm,
    #preprocessors=[getml.preprocessors.Mapping()],
    feature_learners=[getml.feature_learning.FastProp()],
    predictors=[getml.predictors.XGBoostClassifier()],
    loss_function=getml.feature_learning.loss_functions.CrossEntropyLoss,
)
# %%
pipe.fit(container.train)
# %%
print(pipe.score(container.test))
print(pipe.score(container.val))
# %%
