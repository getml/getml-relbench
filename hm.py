# %%

import getml
from relbench.datasets import get_dataset
from relbench.tasks import get_task

dataset = get_dataset("rel-hm", download=True)
task = get_task("rel-hm", "user-churn", download=True)

getml.set_project("rel-hm")

population_roles = getml.data.Roles(
    join_key=["customer_id"],
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

customer_roles = getml.data.Roles(
    join_key=["customer_id"],
    numerical=["age"],
    categorical=[
        "club_member_status",
        "fashion_news_frequency",
    ],
)

customer = getml.DataFrame.from_parquet(
    f"{dataset.cache_dir}/db/customer.parquet",
    name="customer",
    roles=customer_roles,
)

customer.set_unit(["FN", "Active", "postal_code"], "comparison only")

transaction_roles = getml.data.Roles(
    join_key=["article_id", "customer_id"],
    time_stamp=["t_dat"],
    numerical=["price"],
    categorical=["sales_channel_id"],
)

transaction = getml.DataFrame.from_parquet(
    f"{dataset.cache_dir}/db/transactions.parquet",
    name="transaction",
    roles=transaction_roles,
)

article_roles = getml.data.Roles(
    join_key=["article_id"],
    numerical=[
        "product_code",
        "product_type_no",
        "perceived_colour_master_id",
        "department_no",
        "section_no",
    ],
    categorical=[
        "product_type_name",
        "product_group_name",
        "graphical_appearance_no",
        "graphical_appearance_name",
        "colour_group_code",
        "colour_group_name",
        "perceived_colour_value_id",
        "perceived_colour_value_name",
        "perceived_colour_master_name",
        "department_name",
        "index_code",
        "index_name",
        "index_group_no",
        "index_group_name",
        "garment_group_no",
        "garment_group_name",
    ],
    # text=["prod_name", "section_name", "detail_desc"], # broken
)

article = getml.DataFrame.from_parquet(
    f"{dataset.cache_dir}/db/article.parquet",
    name="article",
    roles=article_roles,
)


dm = getml.data.DataModel(population=populations["train"].to_placeholder())
dm.add(
    customer.to_placeholder(), transaction.to_placeholder(), article.to_placeholder()
)
dm.population.join(
    dm.customer, on="customer_id", relationship=getml.data.relationship.many_to_one
)
dm.population.join(
    dm.transaction, on="customer_id", time_stamps=("reference_date", "t_dat")
)
dm.transaction.join(
    dm.article, on="article_id", relationship=getml.data.relationship.many_to_one
)

container = getml.data.Container(**populations)
container.add(customer, transaction, article)

pipe = getml.Pipeline(
    tags=["task: user-churn"],
    data_model=dm,
    feature_learners=[getml.feature_learning.FastProp()],
    predictors=[getml.predictors.XGBoostClassifier()],
    loss_function=getml.feature_learning.loss_functions.CrossEntropyLoss,
)

pipe.fit(container.train)


pipe.score(container.test)

predictions = {}
for subset in subsets:
    predictions[subset] = pipe.predict(container[subset])

task.evaluate(predictions["test"], target_table=task.get_table("test"))
task.evaluate(predictions["val"], target_table=task.get_table("val"))
