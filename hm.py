# %%

import getml
from relbench.datasets import get_dataset
from relbench.tasks import get_task

dataset = get_dataset("rel-hm", download=True)
task = get_task("rel-hm", "user-churn", download=True)

getml.engine.launch(in_memory=False)

getml.set_project("rel-hm")


def load_df(path: str, name: str, roles: getml.data.Roles) -> getml.data.DataFrame:
    """
    Load a DataFrame from a parquet file and save it in getml native format to
    disk if it does not exist yet.
    """
    if getml.data.exists(name):
        return getml.data.load_data_frame(name)

    df = getml.data.DataFrame.from_parquet(
        path,
        name=name,
        roles=roles,
    )
    df.save()
    return df


population_roles = getml.data.Roles(
    join_key=["customer_id"],
    target=["churn"],
    time_stamp=["timestamp"],
)

subsets = ("train", "test", "val")
populations = {}
for subset in subsets:
    populations[subset] = load_df(
        f"{dataset.cache_dir}/tasks/user-churn/{subset}.parquet",
        subset,
        population_roles,
    )

customer_roles = getml.data.Roles(
    join_key=["customer_id"],
    numerical=["age"],
    categorical=[
        "club_member_status",
        "fashion_news_frequency",
        "postal_code",
    ],
)

customer = load_df(
    f"{dataset.cache_dir}/db/customer.parquet", "customer", customer_roles
)
customer.set_unit(["FN", "Active", "postal_code"], "comparison only")

transaction_roles = getml.data.Roles(
    join_key=["article_id", "customer_id"],
    time_stamp=["t_dat"],
    numerical=["price"],
    categorical=["sales_channel_id"],
)

transaction = load_df(
    f"{dataset.cache_dir}/db/transactions.parquet", "transaction", transaction_roles
)

article_roles = getml.data.Roles(
    join_key=["article_id"],
    categorical=[
        # "product_type_name",
        "product_group_name",
        # "graphical_appearance_name",
        # "colour_group_name",
        # "perceived_colour_value_name",
        # "perceived_colour_master_name",
        # "section_no",
        "department_name",
        # "index_name",
        # "index_group_name",
        "garment_group_name",
    ],
    # text=["prod_name", "section_name", "detail_desc"], # broken
)

article = load_df(f"{dataset.cache_dir}/db/article.parquet", "article", article_roles)

dm = getml.data.DataModel(population=populations["train"].to_placeholder())
dm.add(getml.data.to_placeholder(customer, transaction, article))
dm.population.join(
    dm.customer, on="customer_id", relationship=getml.data.relationship.many_to_one
)
dm.population.join(
    dm.transaction,
    on="customer_id",
    time_stamps=("timestamp", "t_dat"),
)
dm.transaction.join(
    dm.article, on="article_id", relationship=getml.data.relationship.many_to_one
)

container = getml.data.Container(**populations)
container.add(customer, transaction, article)

pipe = getml.Pipeline(
    tags=["task: user-churn"],
    data_model=dm,
    preprocessors=[getml.preprocessors.Seasonal(), getml.preprocessors.Mapping()],
    feature_learners=[
        getml.feature_learning.FastProp(
            num_threads=8,
            n_most_frequent=3,
            aggregation=getml.feature_learning.FastProp.agg_sets.default
            | {
                getml.feature_learning.aggregations.COUNT_DISTINCT_OVER_COUNT,
                getml.feature_learning.aggregations.EWMA_1D,
                getml.feature_learning.aggregations.EWMA_7D,
                getml.feature_learning.aggregations.EWMA_30D,
                getml.feature_learning.aggregations.EWMA_90D,
                getml.feature_learning.aggregations.EWMA_365D,
                getml.feature_learning.aggregations.EWMA_TREND_7D,
                getml.feature_learning.aggregations.EWMA_TREND_30D,
                getml.feature_learning.aggregations.EWMA_TREND_90D,
                getml.feature_learning.aggregations.EWMA_TREND_365D,
                getml.feature_learning.aggregations.Q_1,
                getml.feature_learning.aggregations.Q_5,
                getml.feature_learning.aggregations.Q_10,
                getml.feature_learning.aggregations.Q_25,
                getml.feature_learning.aggregations.TIME_SINCE_FIRST_MINIMUM,
                getml.feature_learning.aggregations.TIME_SINCE_LAST_MINIMUM,
                getml.feature_learning.aggregations.TIME_SINCE_LAST_MAXIMUM,
                getml.feature_learning.aggregations.TIME_SINCE_FIRST_MAXIMUM,
            },
        )
    ],
    predictors=[getml.predictors.XGBoostClassifier(n_jobs=8)],
    feature_selectors=[getml.predictors.XGBoostClassifier(n_jobs=8)],
    loss_function=getml.feature_learning.loss_functions.CrossEntropyLoss,
)

pipe.check(container.train)

pipe.fit(container.train)


pipe.score(container.test)

predictions = {}
for subset in subsets:
    predictions[subset] = pipe.predict(container[subset])

task.evaluate(predictions["test"])
task.evaluate(predictions["val"], target_table=task.get_table("val"))
