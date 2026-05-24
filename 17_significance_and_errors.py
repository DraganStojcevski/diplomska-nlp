"""
17_significance_and_errors.py — McNemar significance tests + error analysis.

Two outputs:

  1) Pairwise McNemar tests (two-sided exact binomial) on key model
     pairs, to formalize claims the deck and thesis make:
       - "DistilBERT FT vs FinBERT is statistically a tie."
       - "BERT-large retrain beats DistilBERT FT / FinBERT significantly."
       - "The three top transformers all beat the classical baseline
         significantly."

  2) Error analysis on the three top models (FinBERT, DistilBERT FT,
     BERT-large retrain): per-model error counts, true->predicted
     confusion patterns, and the test sentences that even all three
     models get wrong.

Outputs (all new files):
  results/mcnemar_results.csv
  results/mcnemar_summary.md
  results/error_analysis.csv
  results/error_analysis_summary.md

Does NOT modify any existing files. Operates entirely on saved
prediction CSVs.

Run:
  conda run -n diplomska-nlp python 17_significance_and_errors.py
"""

from pathlib import Path
import pandas as pd
from scipy.stats import binomtest


DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
TARGET_NAMES = ["negative", "neutral", "positive"]

# (display_name, filename, prediction_column)
MODELS = {
    "General Model":      ("general_predictions.csv",                     "general_predicted_label"),
    "FinBERT":            ("finbert_predictions.csv",                     "finbert_predicted_label"),
    "DistilBERT FT":      ("distilbert_finetuned_loaded_predictions.csv", "distilbert_finetuned_predicted_label"),
    "BERT-large FT":      ("bert_large_finetuned_predictions.csv",        "bert_large_finetuned_predicted_label"),
    "BERT-large retrain": ("bert_large_retrain_predictions.csv",          "bert_large_retrain_predicted_label"),
    "Classical baseline": ("classical_baseline_predictions.csv",          "classical_baseline_predicted_label"),
}

# Key pairs that answer specific questions from the deck/thesis.
PAIRS = [
    ("DistilBERT FT",      "FinBERT",            "Are these statistically equivalent? (multi-seed suggested tie)"),
    ("BERT-large retrain", "DistilBERT FT",      "Does corrected BERT-large beat the small fine-tuned model?"),
    ("BERT-large retrain", "FinBERT",            "Does corrected BERT-large beat the domain-pretrained model?"),
    ("BERT-large retrain", "Classical baseline", "Does the best transformer beat the classical floor?"),
    ("DistilBERT FT",      "Classical baseline", "Does fine-tuned DistilBERT beat the classical floor?"),
    ("FinBERT",            "Classical baseline", "Does FinBERT beat the classical floor?"),
]

TOP = ["FinBERT", "DistilBERT FT", "BERT-large retrain"]


def load_predictions():
    by_model = {}
    truth = None
    for name, (filename, column) in MODELS.items():
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"  Skipping {name} -- missing {path}", flush=True)
            continue
        df = pd.read_csv(path)
        df = df.sort_values("row_id").reset_index(drop=True)
        by_model[name] = df[column].tolist()
        if truth is None:
            truth = df["label"].tolist()
    return by_model, truth


def mcnemar_exact(pred_a, pred_b, truth):
    """Two-sided exact binomial McNemar test."""
    b = 0
    c = 0
    for pa, pb, t in zip(pred_a, pred_b, truth):
        if pa == t and pb != t:
            b += 1
        elif pa != t and pb == t:
            c += 1
    n = b + c
    if n == 0:
        return b, c, 1.0
    p_value = binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue
    return b, c, p_value


def format_p(p):
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def write_mcnemar_outputs(rows):
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "mcnemar_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}", flush=True)

    md = [
        "# McNemar Pairwise Tests\n",
        "\n",
        "Two-sided exact binomial McNemar test on the 453-sentence test set.\n",
        "\n",
        "- `b` = #(Model A correct, Model B wrong)\n",
        "- `c` = #(Model A wrong, Model B correct)\n",
        "- p < 0.05 = statistically significant difference between the two classifiers' errors\n",
        "\n",
        "## Results\n",
        "\n",
        "| Pair | b | c | p-value | Significant? | Winner |\n",
        "|------|---:|---:|---:|:---:|:---|\n",
    ]
    for row in rows:
        md.append(
            f"| {row['Model A']} vs {row['Model B']} "
            f"| {row['A right, B wrong (b)']} "
            f"| {row['A wrong, B right (c)']} "
            f"| {format_p(row['p-value'])} "
            f"| {row['Significant (alpha=0.05)']} "
            f"| {row['Winner']} |\n"
        )
    md.append("\n## Questions answered\n\n")
    for row in rows:
        verdict = "yes" if row["Significant (alpha=0.05)"] == "yes" else "no"
        winner = row["Winner"]
        md.append(
            f"- **{row['Question']}**\n"
            f"  - Verdict: {verdict} (p = {format_p(row['p-value'])}), winner: {winner}\n"
        )
    md_path = RESULTS_DIR / "mcnemar_summary.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"Saved: {md_path}", flush=True)


def write_error_analysis_outputs(by_model, truth):
    available_top = [m for m in TOP if m in by_model]
    if not available_top:
        print("No top models available for error analysis -- skipping.", flush=True)
        return

    test_df = pd.read_csv(DATA_DIR / "test.csv").sort_values("row_id").reset_index(drop=True)

    error_rows = []
    for i, row in test_df.iterrows():
        true_label = truth[i]
        per_model_pred = {m: by_model[m][i] for m in available_top}
        wrong_models = [m for m, p in per_model_pred.items() if p != true_label]
        if not wrong_models:
            continue
        error_rows.append({
            "row_id": row["row_id"],
            "sentence": row["sentence"],
            "true_label": TARGET_NAMES[true_label],
            **{f"{m}_pred": TARGET_NAMES[per_model_pred[m]] for m in available_top},
            "wrong_models": ", ".join(wrong_models),
            "num_wrong": len(wrong_models),
        })
    error_df = pd.DataFrame(error_rows).sort_values(
        ["num_wrong", "row_id"], ascending=[False, True]
    )
    csv_path = RESULTS_DIR / "error_analysis.csv"
    error_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}  ({len(error_df)} sentences with at least one wrong prediction)", flush=True)

    per_model_errors = {m: sum(1 for i, t in enumerate(truth) if by_model[m][i] != t) for m in available_top}

    md = [
        "# Error Analysis\n",
        "\n",
        f"Test set size: {len(truth)} sentences. Top three models analyzed: ",
        ", ".join(available_top) + ".\n",
        f"\nTotal sentences mis-classified by at least one of these models: **{len(error_df)}**.\n",
        "\n",
        "## Per-model error counts\n",
        "\n",
        "| Model | Errors | Error rate |\n",
        "|-------|------:|-----------:|\n",
    ]
    for m, n in per_model_errors.items():
        md.append(f"| {m} | {n} | {100 * n / len(truth):.1f}% |\n")

    md.append("\n## Error patterns (true → predicted, top 5 per model)\n\n")
    for m in available_top:
        md.append(f"### {m}\n\n")
        md.append("| True | Predicted | Count |\n|------|-----------|------:|\n")
        confusions = {}
        for i, t in enumerate(truth):
            p = by_model[m][i]
            if p != t:
                key = (TARGET_NAMES[t], TARGET_NAMES[p])
                confusions[key] = confusions.get(key, 0) + 1
        for (true_label, predicted), count in sorted(confusions.items(), key=lambda x: -x[1])[:5]:
            md.append(f"| {true_label} | {predicted} | {count} |\n")
        md.append("\n")

    all_wrong = error_df[error_df["num_wrong"] == len(available_top)]
    md.append(f"## Sentences mis-classified by ALL {len(available_top)} top models\n\n")
    if len(all_wrong) == 0:
        md.append("None! All three models always agree with at least one of them being right.\n")
    else:
        md.append(
            f"These are the hardest sentences in the test set — "
            f"three independent model approaches all fail. ({len(all_wrong)} total; first 10 shown.)\n\n"
        )
        for _, row in all_wrong.head(10).iterrows():
            md.append(f"- **True: {row['true_label']}** — \"{row['sentence']}\"\n")
            for m in available_top:
                md.append(f"  - {m}: predicted *{row[f'{m}_pred']}*\n")
            md.append("\n")

    md_path = RESULTS_DIR / "error_analysis_summary.md"
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"Saved: {md_path}", flush=True)

    print("\nPer-model errors:", flush=True)
    for m, n in per_model_errors.items():
        print(f"  {m}: {n}/{len(truth)} ({100 * n / len(truth):.1f}%)", flush=True)
    print(f"\n  Hardest sentences (all {len(available_top)} top models wrong): {len(all_wrong)}", flush=True)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    print("Loading predictions...", flush=True)
    by_model, truth = load_predictions()
    print(f"Loaded {len(by_model)} models, test set size {len(truth)}.\n", flush=True)

    rows = []
    print("McNemar pairwise tests:", flush=True)
    for a, b_name, question in PAIRS:
        if a not in by_model or b_name not in by_model:
            print(f"  Skipped: {a} vs {b_name} (missing predictions)", flush=True)
            continue
        b, c, p = mcnemar_exact(by_model[a], by_model[b_name], truth)
        winner = a if b > c else (b_name if c > b else "tie")
        significant = "yes" if p < 0.05 else "no"
        rows.append({
            "Model A": a,
            "Model B": b_name,
            "A right, B wrong (b)": b,
            "A wrong, B right (c)": c,
            "Disagreements": b + c,
            "p-value": p,
            "Significant (alpha=0.05)": significant,
            "Winner": winner,
            "Question": question,
        })
        marker = "SIG" if significant == "yes" else "ns "
        print(
            f"  [{marker}] {a} vs {b_name}: b={b}, c={c}, "
            f"p={format_p(p)}, winner={winner}",
            flush=True,
        )

    print(flush=True)
    write_mcnemar_outputs(rows)
    write_error_analysis_outputs(by_model, truth)


if __name__ == "__main__":
    main()
