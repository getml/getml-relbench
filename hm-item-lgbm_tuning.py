import gc
import logging
from pathlib import Path

import lightgbm as lgb
import optuna
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")

FEATURES_PARQUET_PATH_TEMPLATE = "hm_item_pipe_refined_{subset}_features.parquet"
FEATURES_LGBM_BIN_PATH_TEMPLATE = "hm_item_pipe_refined_{subset}_features.bin"
FEATURES_DROP_COLS = ["sales", "article_id", "timestamp"]

LGBM_NUM_BOOST_ROUND = 2000
LGBM_EARLY_STOPPING_ROUNDS = 50
LGBM_LOG_EVALUATION_PERIOD = 0
LGBM_VERBOSE_EVAL = False

OPTUNA_STUDY_NAME = "opt-hm-item"
OPTUNA_STORAGE_URL = f"sqlite:///{OPTUNA_STUDY_NAME}.db"

################################################################################
# 1. Load data and preprocess
# ##############################################################################

train_arrow = pq.read_table(
    FEATURES_PARQUET_PATH_TEMPLATE.format(subset="train"), memory_map=True
)
val_arrow = pq.read_table(
    FEATURES_PARQUET_PATH_TEMPLATE.format(subset="val"), memory_map=True
)
test_arrow = pq.read_table(
    FEATURES_PARQUET_PATH_TEMPLATE.format(subset="test"), memory_map=True
)


def encode_categoricals(table):
    categical_cols = []
    for name in table.column_names:
        if pa.types.is_dictionary(table.column(name).type):
            table = table.append_column(
                f"{name}_encoded",
                pa.chunked_array(chunk.indices for chunk in table.column(name).chunks),
            )
            categical_cols.append(f"{name}_encoded")
    return table, categical_cols


def select_columns(table):
    return table.select(
        [
            name
            for name in table.column_names
            if (
                pa.types.is_integer(table.column(name).type)
                or pa.types.is_floating(table.column(name).type)
            )
            and name not in FEATURES_DROP_COLS
        ]
    )


X_train, categorical_cols = encode_categoricals(select_columns(train_arrow))
y_train = train_arrow["sales"]

X_val, _ = encode_categoricals(select_columns(val_arrow))
y_val = val_arrow["sales"]

X_test, _ = encode_categoricals(select_columns(test_arrow))
y_test = test_arrow["sales"]


###############################################################################
# 2. Create LightGBM datasets
###############################################################################


def create_binary_dataset(X, y, categorical_cols, file_name, reference=None):
    """
    Create a LightGBM dataset as a binary dataset file. Manually free up memory
    and return a pointer to the dataset file.
    """
    dataset = lgb.Dataset(
        X,
        label=y,
        categorical_feature=categorical_cols,
        free_raw_data=False,
        reference=reference,
    )
    path = Path(file_name)
    if path.exists():
        path.unlink()
    dataset.save_binary(path)
    del dataset
    gc.collect()
    return lgb.Dataset(path, free_raw_data=False)


train = create_binary_dataset(
    X_train,
    y_train,
    categorical_cols,
    FEATURES_LGBM_BIN_PATH_TEMPLATE.format(subset="train"),
)
val = create_binary_dataset(
    X_val,
    y_val,
    categorical_cols,
    FEATURES_LGBM_BIN_PATH_TEMPLATE.format(subset="val"),
    reference=train,
)


###############################################################################
# 3. Define objective function
# ##############################################################################


def objective(trial):
    params = {
        "two_round": False,  # enable two_round loading for dataset two save memory
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "bagging_freq": 1,
        "feature_pre_filter": False,
        "max_depth": trial.suggest_int("max_depth", 3, 11),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 1024),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-9, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-9, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = lgb.train(
        params,
        train,
        num_boost_round=LGBM_NUM_BOOST_ROUND,
        valid_sets=[val],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS, verbose=True
            ),
            lgb.log_evaluation(period=LGBM_LOG_EVALUATION_PERIOD),
        ],
    )

    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    mae_val = mean_absolute_error(y_val, pred_val)

    logger.info(
        f"Trial {trial.number} finished with MAE={mae_val:.6f}, params={trial.params}"
    )

    return mae_val


###############################################################################
# 4. Create or load an Optuna study from SQLite for resuming
# ##############################################################################

study = optuna.create_study(
    study_name=OPTUNA_STUDY_NAME,
    storage=OPTUNA_STORAGE_URL,
    load_if_exists=True,
    direction="minimize",
)
logger.info(f"Starting Optuna study {OPTUNA_STUDY_NAME!r}...")

###############################################################################
# 5. Run optimization
# ##############################################################################

# study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params["objective"] = "regression_l1"
best_params["metric"] = "mae"
best_params["feature_pre_filter"] = False

logger.info(f"Best trial found: {study.best_trial.number}")
logger.info(f"Best params: {best_params}")

###############################################################################
# 6. Final model training with best parameters
# ##############################################################################

final_model = lgb.train(
    best_params,
    train,
    num_boost_round=LGBM_NUM_BOOST_ROUND,
    valid_sets=[val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=LGBM_LOG_EVALUATION_PERIOD),
    ],
)

pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
test_mae = mean_absolute_error(y_test, pred_test)

pred_val = final_model.predict(X_val, num_iteration=final_model.best_iteration)
val_mae = mean_absolute_error(y_val, pred_val)

pred_train = final_model.predict(X_train, num_iteration=final_model.best_iteration)
train_mae = mean_absolute_error(y_train, pred_train)


# Print final result to log
logger.info(f"Final model Train MAE: {train_mae:.6f}")
logger.info(f"Final model Val MAE: {val_mae:.6f}")
logger.info(f"Final model Test MAE: {test_mae:.6f}")

###############################################################################
# 7. Script end
# ##############################################################################
logger.info("Study completed.")
