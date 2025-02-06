import logging
import pandas as pd
import lightgbm as lgb
import optuna

# Import roc_auc_score instead of auc
from sklearn.metrics import roc_auc_score

# optuna_hm-churn.py >> opt-hm-churn.log 2>&1 &

pref = "_200"


##############################################################################
# 1. Set up dedicated logging
# #############################################################################

logging.basicConfig(
    filename=f"opt-az-item-churn{pref}.log",    # File where all logs will be written
    level=logging.INFO,             # You can switch to DEBUG for more detail
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

###############################################################################
# 2. Load and preprocess data
# ##############################################################################

train_df = pd.read_parquet(f"train_transform{pref}")
val_df = pd.read_parquet(f"val_transform{pref}")
test_df = pd.read_parquet(f"test_transform{pref}")

X_train = train_df.drop(columns=["churn"])
y_train = train_df["churn"]

X_val = val_df.drop(columns=["churn"])
y_val = val_df["churn"]

X_test = test_df.drop(columns=["churn"])
y_test = test_df["churn"]


def convert_object_columns(df):
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = df[col].astype(int)
        except (ValueError, TypeError):
            df[col] = df[col].astype("category")
    return df

X_train = convert_object_columns(X_train)
X_val   = convert_object_columns(X_val)
X_test  = convert_object_columns(X_test)

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
# 3. Objective function (Optuna)
# ##############################################################################
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,               # Suppress LightGBM's default console spam
        "bagging_freq": 1,
        "feature_pre_filter": False,
        # "num_threads": 32,
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
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),  # period=0 suppresses iteration logs
        ]
    )

    # Predict on validation set
    pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    auc_val = roc_auc_score(y_val, pred_val)

    # Log the result for this trial
    logger.info(
        f"Trial {trial.number} finished with AUROC={auc_val:.6f}, params={trial.params}"
    )

    return auc_val

###############################################################################
# 4. Create or load Optuna study using SQLite
# ##############################################################################
storage_url = f"sqlite:///opt-az-item-churn{pref}.db"
study_name  = f"opt-az-item-churn{pref}"

study = optuna.create_study(
    study_name=study_name,
    storage=storage_url,
    load_if_exists=True,
    direction="maximize",
)
logger.info("Starting Optuna study...")

###############################################################################
# 5. Run optimization
# ##############################################################################
study.optimize(objective, n_trials=50)

best_params = study.best_params
best_params["objective"] = "binary"
best_params["metric"] = "auc"
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
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=0),  # No iteration logs, only final result
    ]
)

pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
test_auc = roc_auc_score(y_test, pred_test)
logger.info(f"Final model Test AUC: {test_auc:.6f}")

pred_train = final_model.predict(X_train, num_iteration=final_model.best_iteration)
train_auc = roc_auc_score(y_train, pred_train)
logger.info(f"Final model Train AUC: {train_auc:.6f}")

pred_val = final_model.predict(X_val, num_iteration=final_model.best_iteration)
val_auc = roc_auc_score(y_val, pred_val)
logger.info(f"Final model Val AUC: {val_auc:.6f}")

pred_test = final_model.predict(X_test, num_iteration=final_model.best_iteration)
test_auc = roc_auc_score(y_test, pred_test)
logger.info(f"Final model Test AUC: {test_auc:.6f}")

###############################################################################
# 7. Script end
# ##############################################################################
logger.info("Study completed.")
