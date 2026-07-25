import pickle

import matplotlib.pyplot as plt
import shap
from pyspark.sql import SparkSession
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from modeling import config, utils


def write_reports(model, X_train, X_test, metrics: dict):
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with open(config.REPORT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(config.REPORT_DIR / "metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for ax, col in zip(axes.flat, config.FEATURES):
        ax.hist(X_train[col], bins=30)
        ax.set_title(col, fontsize=9)
    plt.tight_layout()
    fig.savefig(config.REPORT_DIR / "feature_histograms.png")
    plt.close(fig)

    top_features = [f for _, f in sorted(zip(model.feature_importances_, config.FEATURES), reverse=True)[:4]]
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(model, X_train, top_features, ax=ax)
    fig.savefig(config.REPORT_DIR / "pdp.png")
    plt.close(fig)

    shap_values = shap.TreeExplainer(model).shap_values(X_test)
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    fig.savefig(config.REPORT_DIR / "shap_summary.png", bbox_inches="tight")
    plt.close(fig)


def main():
    for d in (config.RAW_DIR, config.PROCESSED_DIR, config.REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    calls = utils.fetch_calls_weekly(config.START_DATE, config.END_DATE)
    events = utils.fetch_events_weekly(config.START_DATE, config.END_DATE)
    lag1_weather, pred_weather = utils.fetch_weather_weekly(config.START_DATE, config.END_DATE)

    calls.to_parquet(config.RAW_DIR / "calls_weekly.parquet")
    events.to_parquet(config.RAW_DIR / "events_weekly.parquet")
    lag1_weather.to_parquet(config.RAW_DIR / "weather_lag1.parquet")
    pred_weather.to_parquet(config.RAW_DIR / "weather_pred.parquet")

    spark = SparkSession.builder.appName("modeling-train").master("local[*]").getOrCreate()
    data = utils.build_features(spark, calls, events, lag1_weather, pred_weather)
    spark.stop()
    data[config.FEATURES] = data[config.FEATURES].astype(float)
    data.to_parquet(config.PROCESSED_DIR / "features.parquet")

    train, test = utils.chronological_split(data)
    X_train, y_train = train[config.FEATURES], train[config.TARGET]
    X_test, y_test = test[config.FEATURES], test[config.TARGET]

    model = XGBRegressor(**config.MODEL_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "baseline_mae": mean_absolute_error(y_test, X_test["lag_1"]),
    }
    write_reports(model, X_train, X_test, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
