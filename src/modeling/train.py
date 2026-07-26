import pickle
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shap
from pyspark.sql import SparkSession
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from modeling import config, utils

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "00_raw"
PROCESSED_DIR = ROOT / "data" / "01_processed"
REPORT_DIR = ROOT / "data" / "02_reporting"

RANKING_K = 5


def evaluate(train, test):
    X_train, y_train = train[config.FEATURES], train[config.TARGET]
    X_test, y_test = test[config.FEATURES], test[config.TARGET]

    model = XGBRegressor(**config.MODEL_PARAMS)
    model.fit(X_train, np.log1p(y_train))
    pred = np.expm1(model.predict(X_test))

    metrics = {
        "mae": mean_absolute_error(y_test, pred),
        "rmse": mean_squared_error(y_test, pred) ** 0.5,
        "baseline_mae": mean_absolute_error(y_test, test["lag_1_raw"]),
    }

    test = test.copy()
    test["pred"] = pred
    metrics.update({f"model_{k}": v for k, v in utils.ranking_metrics(test, "pred", RANKING_K).items()})
    metrics.update({f"baseline_{k}": v for k, v in utils.ranking_metrics(test, "lag_1_raw", RANKING_K).items()})

    return metrics, model, X_train, X_test


def write_metrics(all_metrics: dict[str, dict]):
    with open(REPORT_DIR / "metrics.txt", "w") as f:
        for section, metrics in all_metrics.items():
            f.write(f"=== {section} ===\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v:.4f}\n")
            f.write("\n")


def plot_feature_histograms_pdp(model, X_train, path):
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, col in zip(axes.flat, config.FEATURES):
        ax.hist(X_train[col], bins=30, color="#93c5fd", alpha=0.7)
        ax.set_title(col, fontsize=9)
        pdp = partial_dependence(model, X_train, [col], kind="average")
        ax2 = ax.twinx()
        ax2.plot(pdp["grid_values"][0], pdp["average"][0], color="tomato", linewidth=2)
        ax2.set_yticks([])
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_shap(model, X_test, prefix: Path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    fig.savefig(f"{prefix}_summary.png", bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    fig.savefig(f"{prefix}_bar.png", bbox_inches="tight")
    plt.close(fig)

    top_features = [f for _, f in sorted(zip(model.feature_importances_, config.FEATURES), reverse=True)[:4]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, feat in zip(axes.flat, top_features):
        shap.dependence_plot(feat, shap_values, X_test, ax=ax, show=False)
    plt.tight_layout()
    fig.savefig(f"{prefix}_dependence.png")
    plt.close(fig)


def plot_calls_by_district(calls, path):
    totals = calls.groupby("board_key")["calls"].sum().reset_index(name="total_calls")
    districts = gpd.read_file(f"{config.DISTRICTS_URL}?$limit=500")
    districts["board_key"] = districts["boro_cd"].apply(
        lambda s: s[1:].zfill(2) + " " + config.BORO_NAMES[s[0]]
    )
    merged = districts.merge(totals, on="board_key", how="left")

    fig, ax = plt.subplots(figsize=(10, 10))
    merged.plot(column="total_calls", cmap="viridis", legend=True, ax=ax,
               edgecolor="black", linewidth=0.3, missing_kwds={"color": "lightgrey"})
    ax.set_axis_off()
    ax.set_title("Total 311 calls by community district")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    for d in (RAW_DIR, PROCESSED_DIR, REPORT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    calls = utils.fetch_calls_weekly(config.START_DATE, config.END_DATE)
    events = utils.fetch_events_weekly(config.START_DATE, config.END_DATE)
    lag1_weather, pred_weather = utils.fetch_weather_weekly(config.START_DATE, config.END_DATE)

    calls.to_parquet(RAW_DIR / "calls_weekly.parquet")
    events.to_parquet(RAW_DIR / "events_weekly.parquet")
    lag1_weather.to_parquet(RAW_DIR / "weather_lag1.parquet")
    pred_weather.to_parquet(RAW_DIR / "weather_pred.parquet")

    spark = SparkSession.builder.appName("modeling-train").master("local[*]").getOrCreate()
    data = utils.build_features(spark, calls, events, lag1_weather, pred_weather)
    spark.stop()
    data = utils.prepare_features(data)
    data.to_parquet(PROCESSED_DIR / "features.parquet")

    chrono_train, chrono_test = utils.chronological_split(data)
    chrono_metrics, model, X_train, X_test = evaluate(chrono_train, chrono_test)

    random_train, random_test = train_test_split(data, test_size=0.2, random_state=42)
    random_metrics, _, _, _ = evaluate(random_train, random_test)

    board_train, board_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["board_key"])
    board_metrics, _, _, _ = evaluate(board_train, board_test)

    week_train, week_test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["week_start"])
    week_metrics, _, _, _ = evaluate(week_train, week_test)

    write_metrics({
        "chronological split (production model)": chrono_metrics,
        "random split": random_metrics,
        "stratified split (community board)": board_metrics,
        "stratified split (week)": week_metrics,
    })

    with open(REPORT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    plot_feature_histograms_pdp(model, X_train, REPORT_DIR / "feature_histograms_pdp.png")
    plot_shap(model, X_test, REPORT_DIR / "shap")
    plot_calls_by_district(calls, REPORT_DIR / "calls_by_district_map.png")

    print(chrono_metrics)


if __name__ == "__main__":
    main()
