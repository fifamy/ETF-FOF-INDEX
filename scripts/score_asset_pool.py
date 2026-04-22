#!/usr/bin/env python3
import argparse
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

from etf_fof_index.scoring import load_candidate_pool, load_scoring_config, score_candidate_pool  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score ETF asset-pool candidates by bucket.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "asset_pool_scoring_v1.yaml"),
        help="Path to scoring YAML config.",
    )
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "asset_pool_candidates_scoring_v1.csv"),
        help="Candidate pool CSV with raw metrics.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "output" / "asset_pool_scores_v1.csv"),
        help="Scored CSV output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_scoring_config(Path(args.config))
    pool = load_candidate_pool(Path(args.input))
    scored = score_candidate_pool(pool, config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)

    print(f"Scored rows: {len(scored)}")
    print(f"Output written to: {output_path}")
    passed = scored[scored["hard_filter_pass"]]
    if not passed.empty:
        print("\nTop candidates by bucket:")
        summary = passed.sort_values(["bucket", "bucket_rank"]).groupby("bucket").head(3)
        for _, row in summary.iterrows():
            print(
                f"- {row['bucket']} | rank {int(row['bucket_rank'])} | {row['symbol']} | "
                f"score {row['final_score']:.2f} | group {row.get('benchmark_group', '')}"
            )


if __name__ == "__main__":
    main()

