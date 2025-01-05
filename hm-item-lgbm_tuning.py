import sys
import logging
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_absolute_error

##############################################################################
# 1. Set up dedicated logging
# #############################################################################
logging.basicConfig(
    filename="opt-hm-item.log",    # File where all logs will be written
    level=logging.INFO,             # You can switch to DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


###############################################################################
# 2. Load data and preprocess
# ##############################################################################
train_df = pd.read_parquet("train_features_final-hm-item.parquet")
val_df = pd.read_parquet("val_features_final-hm-item.parquet")
test_df = pd.read_parquet("test_features_final-hm-item.parquet")

X_train = train_df.drop(columns=["sales", "article_id", "timestamp"])
y_train = train_df["sales"]

X_val = val_df.drop(columns=["sales", "article_id", "timestamp"])
y_val = val_df["sales"]

X_test = test_df.drop(columns=["sales", "article_id", "timestamp"])
y_test = test_df["sales"]

def convert_object_columns(df):
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = df[col].astype(int)
        except (ValueError, TypeError):
            df[col] = df[col].astype("category")
    return df

X_train = convert_object_columns(X_train)
X_val = convert_object_columns(X_val)
X_test = convert_object_columns(X_test)

categorical_cols = X_train.select_dtypes(include=["category"]).columns.tolist()

train_data = lgb.Dataset(
    X_train, label=y_train, 
    categorical_feature=categorical_cols,
    free_raw_data=False
)
val_data = lgb.Dataset(
    X_val, label=y_val, 
    categorical_feature=categorical_cols,
    free_raw_data=False
)

###############################################################################
# 3. Define objective function
# ##############################################################################
def objective(trial):
    params = {
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
        train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[
            # Verbose stopping feedback
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            # Log evaluation every x rounds
            lgb.log_evaluation(period=0), # period=0 suppresses iteration logs
        ]
    )

    # Predict on validation set
    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    mae_val = mean_absolute_error(y_val, pred_val)

    # Log the result for this trial
    logger.info(
        f"Trial {trial.number} finished with MAE={mae_val:.6f}, params={trial.params}"
    )
    
    return mae_val

###############################################################################
# 4. Create or load an Optuna study from SQLite for resuming
# ##############################################################################
storage_url = "sqlite:///opt-hm-item.db"
study_name = "opt-hm-item"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_url,
    load_if_exists=True,
    direction="minimize",
)
logger.info("Starting Optuna study...")

###############################################################################
# 5. Run optimization
# ##############################################################################
study.optimize(objective, n_trials=50)

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
    train_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=0),  # No iteration logs, only final result
    ]
)

pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
test_mae = mean_absolute_error(y_test, pred_test)

# Print final result to log
logger.info(f"Final model Test MAE: {test_mae:.6f}")

###############################################################################
# 7. Script end
# ##############################################################################
logger.info("Study completed.")
