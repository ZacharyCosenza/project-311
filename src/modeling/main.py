import argparse
import pickle

import pandas as pd

from modeling import pipeline_features, pipeline_modeling, pipeline_raw, pipeline_target


def run_train():
    pipeline_modeling.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_raw.run()
    pipeline_target.run()
    features = pipeline_features.run()

    model, full_df = pipeline_modeling.training(features)
    with open(pipeline_modeling.REPORT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    full_df.to_parquet(pipeline_modeling.REPORT_DIR / "modeling_data.parquet")

    all_metrics = pipeline_modeling.metrics(model, full_df)
    pipeline_modeling.write_metrics_csv(all_metrics, pipeline_modeling.REPORT_DIR / "metrics.csv")

    train_df = full_df[full_df[pipeline_modeling.SPLIT_COL] == "train"]
    pipeline_modeling.plot_feature_histograms(
        model, train_df, pipeline_modeling.REPORT_DIR / "feature_histograms.png"
    )

    print(f"model saved to {pipeline_modeling.REPORT_DIR / 'model.pkl'}")


def run_test(model_path: str, data_path: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    df = pd.read_parquet(data_path)
    scored = pipeline_modeling.inference(model, df)

    out_path = pipeline_modeling.REPORT_DIR / "inference_output.parquet"
    scored.to_parquet(out_path)
    print(f"inference output saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("train")
    test_parser = sub.add_parser("test")
    test_parser.add_argument("--model", required=True, help="path to model.pkl")
    test_parser.add_argument("--data", required=True, help="path to a parquet dataframe with FEATURES columns")

    args = parser.parse_args()
    if args.mode == "train":
        run_train()
    else:
        run_test(args.model, args.data)


if __name__ == "__main__":
    main()
