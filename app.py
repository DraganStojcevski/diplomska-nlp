from pathlib import Path

import html

import pandas as pd
import streamlit as st
import torch
from transformers import pipeline


ROOT_DIR = Path(__file__).parent
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

MODEL_COMPARISON_PATH = RESULTS_DIR / "model_comparison.csv"
EPOCH_EXPERIMENT_PATH = RESULTS_DIR / "experiment_epochs.csv"
DATA_SIZE_EXPERIMENT_PATH = RESULTS_DIR / "experiment_data_size.csv"
COMBINED_PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
DISTILBERT_PREDICTIONS_PATH = RESULTS_DIR / "distilbert_finetuned_loaded_predictions.csv"
BERT_LARGE_PREDICTIONS_PATH = RESULTS_DIR / "bert_large_finetuned_predictions.csv"
ANALYSIS_SUMMARY_PATH = RESULTS_DIR / "analysis_summary.md"
DISTILBERT_FINETUNED_DIR = MODELS_DIR / "distilbert-finetuned"
BERT_LARGE_FINETUNED_DIR = MODELS_DIR / "bert-large-finetuned"

FINBERT_MODEL = "ProsusAI/finbert"
PRODUCT_NAME = "Financial Sentiment Intelligence"

DISPLAY_NAMES = {
    "FinBERT": "FinBERT",
    "Fine-tuned DistilBERT": "DistilBERT FT",
    "Fine-tuned BERT-large": "BERT-large",
}

LABEL_NORMALIZATION = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "POSITIVE": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
}

MODEL_DESCRIPTIONS = {
    "FinBERT": "Domain-specific model for financial language.",
    "Fine-tuned DistilBERT": "Small model fine-tuned on Financial PhraseBank.",
    "Fine-tuned BERT-large": "Large model fine-tuned on the same dataset.",
}

PRODUCT_USE_CASES = [
    {
        "title": "News Monitoring",
        "text": "Classify financial news headlines and earnings updates by sentiment.",
    },
    {
        "title": "Analyst Workflow",
        "text": "Flag negative or low-confidence text for manual review.",
    },
    {
        "title": "Model Governance",
        "text": "Compare small, large, and domain-specific models before deployment.",
    },
]

BATCH_EXAMPLES = [
    "The company reported strong revenue growth.",
    "The company announced major losses.",
    "The company will hold its annual meeting next week.",
    "Operating profit increased compared with the previous quarter.",
    "The firm warned that demand may weaken next year.",
]

LABEL_STYLES = {
    "positive": {
        "background": "#dcfce7",
        "border": "#16a34a",
        "text": "#166534",
    },
    "negative": {
        "background": "#fee2e2",
        "border": "#dc2626",
        "text": "#991b1b",
    },
    "neutral": {
        "background": "#e5e7eb",
        "border": "#6b7280",
        "text": "#374151",
    },
}


def inject_styles():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            .hero-panel {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 1.15rem 1.25rem;
                background: linear-gradient(135deg, #f8fafc 0%, #eef6ff 55%, #f0fdf4 100%);
                margin-bottom: 1rem;
            }

            .hero-title {
                color: #0f172a;
                font-size: 1.05rem;
                font-weight: 700;
                margin-bottom: .25rem;
            }

            .hero-copy {
                color: #475569;
                font-size: .95rem;
                margin: 0;
            }

            .product-card {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 1rem;
                background: #ffffff;
                min-height: 132px;
            }

            .product-card-title {
                color: #0f172a;
                font-weight: 700;
                margin-bottom: .35rem;
            }

            .product-card-copy {
                color: #64748b;
                font-size: .9rem;
                margin: 0;
            }

            .model-card {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 1rem;
                background: #ffffff;
                box-shadow: 0 1px 2px rgba(15, 23, 42, .06);
                min-height: 178px;
            }

            .model-title {
                color: #0f172a;
                font-weight: 700;
                font-size: 1rem;
                margin-bottom: .25rem;
            }

            .model-description {
                color: #64748b;
                font-size: .86rem;
                min-height: 40px;
                margin-bottom: .8rem;
            }

            .confidence-value {
                color: #0f172a;
                font-size: 1.6rem;
                font-weight: 700;
                line-height: 1.1;
                margin-top: .7rem;
            }

            .confidence-caption {
                color: #64748b;
                font-size: .82rem;
            }

            .label-badge {
                display: inline-block;
                border-radius: 999px;
                padding: .22rem .6rem;
                font-size: .78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: .02em;
                border: 1px solid;
            }

            .insight-box {
                border-left: 4px solid #2563eb;
                border-radius: 6px;
                background: #eff6ff;
                color: #1e3a8a;
                padding: .75rem .95rem;
                margin: .8rem 0 1rem;
            }

            .warning-box {
                border-left: 4px solid #f59e0b;
                border-radius: 6px;
                background: #fffbeb;
                color: #92400e;
                padding: .75rem .95rem;
                margin: .8rem 0 1rem;
            }

            .small-muted {
                color: #64748b;
                font-size: .9rem;
            }

            .report-line {
                border-bottom: 1px solid #e5e7eb;
                padding: .45rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def saved_model_exists(model_dir):
    return (
        model_dir.exists()
        and (model_dir / "config.json").exists()
        and (
            (model_dir / "model.safetensors").exists()
            or (model_dir / "pytorch_model.bin").exists()
        )
    )


def normalize_label(label):
    return LABEL_NORMALIZATION.get(label, LABEL_NORMALIZATION.get(label.lower(), label))


def label_badge(label):
    normalized_label = normalize_label(label)
    style = LABEL_STYLES.get(normalized_label, LABEL_STYLES["neutral"])

    return (
        "<span class='label-badge' "
        f"style='background:{style['background']};"
        f"border-color:{style['border']};"
        f"color:{style['text']};'>"
        f"{html.escape(normalized_label)}</span>"
    )


def get_pipeline_device():
    if torch.cuda.is_available():
        return 0
    return -1


@st.cache_resource(show_spinner=False)
def load_classifier(model_name):
    if model_name == "FinBERT":
        return pipeline(
            "sentiment-analysis",
            model=FINBERT_MODEL,
            device=get_pipeline_device(),
        )

    if model_name == "Fine-tuned DistilBERT":
        return pipeline(
            "text-classification",
            model=str(DISTILBERT_FINETUNED_DIR),
            tokenizer=str(DISTILBERT_FINETUNED_DIR),
            device=get_pipeline_device(),
        )

    if model_name == "Fine-tuned BERT-large":
        return pipeline(
            "text-classification",
            model=str(BERT_LARGE_FINETUNED_DIR),
            tokenizer=str(BERT_LARGE_FINETUNED_DIR),
            device=get_pipeline_device(),
        )

    raise ValueError(f"Unknown model: {model_name}")


def available_live_models():
    models = ["FinBERT"]

    if saved_model_exists(DISTILBERT_FINETUNED_DIR):
        models.append("Fine-tuned DistilBERT")

    if saved_model_exists(BERT_LARGE_FINETUNED_DIR):
        models.append("Fine-tuned BERT-large")

    return models


def predict_sentiment(model_name, text):
    classifier = load_classifier(model_name)
    result = classifier(text, truncation=True, max_length=128)[0]

    return {
        "Model": DISPLAY_NAMES[model_name],
        "Label": normalize_label(result["label"]),
        "Confidence": result["score"],
    }


def predictions_for_text(model_names, text):
    rows = []

    for model_name in model_names:
        with st.spinner(f"Running {model_name}..."):
            prediction = predict_sentiment(model_name, text)
            prediction["Full Model"] = model_name
            rows.append(prediction)

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_csv(path):
    path = Path(path)

    if not path.exists():
        return None

    return pd.read_csv(path)


def show_table(title, path, missing_message, max_rows=None):
    st.subheader(title)
    data = load_csv(path)

    if data is None:
        st.warning(missing_message)
        return

    shown_data = data if max_rows is None else data.head(max_rows)
    st.dataframe(shown_data, use_container_width=True, hide_index=True)


def show_image_grid(title, graph_paths, columns_per_row=3):
    existing_graphs = [Path(path) for path in graph_paths if Path(path).exists()]

    if not existing_graphs:
        st.warning(f"No graphs found for {title}.")
        return

    st.subheader(title)

    for start in range(0, len(existing_graphs), columns_per_row):
        row_graphs = existing_graphs[start : start + columns_per_row]
        columns = st.columns(len(row_graphs))

        for column, graph_path in zip(columns, row_graphs):
            image_bytes = graph_path.read_bytes()
            modified_time = graph_path.stat().st_mtime
            column.image(
                image_bytes,
                caption=f"{graph_path.name} | updated {modified_time:.0f}",
            )


def show_prediction_cards(predictions_df, confidence_threshold):
    columns = st.columns(len(predictions_df))

    for column, row in zip(columns, predictions_df.to_dict("records")):
        model_name = row["Full Model"]
        display_name = row["Model"]
        label = row["Label"]
        confidence = float(row["Confidence"])
        confidence_percent = confidence * 100
        confidence_note = (
            "Strong prediction"
            if confidence >= confidence_threshold
            else "Needs interpretation"
        )

        column.markdown(
            f"""
            <div class="model-card">
                <div class="model-title">{html.escape(display_name)}</div>
                <div class="model-description">
                    {html.escape(MODEL_DESCRIPTIONS.get(model_name, ""))}
                </div>
                {label_badge(label)}
                <div class="confidence-value">{confidence_percent:.1f}%</div>
                <div class="confidence-caption">{confidence_note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        column.progress(min(max(confidence, 0.0), 1.0))


def get_consensus(predictions_df):
    label_counts = predictions_df["Label"].value_counts()
    top_label = label_counts.index[0]
    top_count = int(label_counts.iloc[0])
    total_count = len(predictions_df)
    avg_confidence = float(predictions_df["Confidence"].mean())

    return top_label, top_count, total_count, avg_confidence


def show_agreement_summary(predictions_df):
    label_counts = predictions_df["Label"].value_counts()

    if len(label_counts) == 1:
        agreed_label = label_counts.index[0]
        st.markdown(
            f"""
            <div class="insight-box">
                All live models agree on the sentiment: {label_badge(agreed_label)}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    summary = ", ".join(
        f"{count} model{'s' if count != 1 else ''} predicted {label}"
        for label, count in label_counts.items()
    )
    st.markdown(
        f"""
        <div class="warning-box">
            The models disagree. This is useful during the defense because it shows why
            model comparison matters: {html.escape(summary)}.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_business_report(predictions_df):
    consensus_label, top_count, total_count, avg_confidence = get_consensus(predictions_df)
    disagreement = top_count != total_count

    confidence_message = (
        "High confidence"
        if avg_confidence >= 0.75
        else "Moderate confidence"
        if avg_confidence >= 0.60
        else "Low confidence"
    )
    review_message = (
        "Send this text to manual review because models disagree."
        if disagreement
        else "This result can be used as an automated sentiment signal."
    )

    st.subheader("Analyst Summary")
    st.markdown(
        f"""
        <div class="insight-box">
            <div class="report-line"><strong>Consensus sentiment:</strong> {label_badge(consensus_label)}</div>
            <div class="report-line"><strong>Model agreement:</strong> {top_count}/{total_count} models</div>
            <div class="report-line"><strong>Average confidence:</strong> {avg_confidence:.3f} ({confidence_message})</div>
            <div><strong>Suggested action:</strong> {html.escape(review_message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_results_overview(results_df):
    if results_df is None or results_df.empty:
        return

    metric_columns = st.columns(3)

    best_accuracy = results_df.loc[results_df["Accuracy"].idxmax()]
    best_f1 = results_df.loc[results_df["F1-score"].idxmax()]

    time_column = (
        "Evaluation Time (seconds)"
        if "Evaluation Time (seconds)" in results_df.columns
        else "Time (seconds)"
    )
    fastest = results_df.loc[results_df[time_column].idxmin()]

    metric_columns[0].metric(
        "Best Accuracy",
        f"{best_accuracy['Accuracy']:.3f}",
        best_accuracy["Model"],
    )
    metric_columns[1].metric(
        "Best F1-score",
        f"{best_f1['F1-score']:.3f}",
        best_f1["Model"],
    )
    metric_columns[2].metric(
        "Fastest Evaluation",
        f"{fastest[time_column]:.2f}s",
        fastest["Model"],
    )


def show_product_dashboard(results_df, models):
    st.markdown(
        f"""
        <div class="hero-panel">
            <div class="hero-title">{PRODUCT_NAME}</div>
            <p class="hero-copy">
                A working prototype for classifying financial text sentiment, comparing model
                quality, and exporting results for analyst review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if results_df is not None and not results_df.empty:
        show_results_overview(results_df)
    else:
        columns = st.columns(3)
        columns[0].metric("Live Models", len(models))
        columns[1].metric("Saved Results", "Missing")
        columns[2].metric("Demo Status", "Ready")

    use_case_columns = st.columns(len(PRODUCT_USE_CASES))
    for column, use_case in zip(use_case_columns, PRODUCT_USE_CASES):
        column.markdown(
            f"""
            <div class="product-card">
                <div class="product-card-title">{html.escape(use_case["title"])}</div>
                <p class="product-card-copy">{html.escape(use_case["text"])}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Why This Is Sellable")
    st.write(
        "The value is not only the model accuracy. The product gives a repeatable workflow: "
        "input financial text, compare several NLP models, detect uncertainty, export results, "
        "and explain why domain-specific models can outperform larger general models."
    )

    st.subheader("Recommended Demo Flow")
    st.markdown(
        """
        1. Start with one positive, one negative, and one neutral sentence in the Live Demo.
        2. Show when models agree and when they disagree.
        3. Open Batch Analyzer and process multiple finance sentences together.
        4. Finish with Final Comparison to prove the research results behind the product.
        """
    )


def show_markdown_file(title, path):
    path = Path(path)
    st.subheader(title)

    if not path.exists():
        st.warning(f"Missing file: {path}")
        return

    st.markdown(path.read_text(encoding="utf-8"))


def show_live_demo(models, example_sentences):
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">Live financial sentiment comparison</div>
            <p class="hero-copy">
                One sentence is sent to every available model, then the app shows
                agreement, confidence, and model differences side by side.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_column, settings_column = st.columns([2, 1])

    with control_column:
        selected_example = st.selectbox(
            "Demo sentence",
            ["Custom sentence", *example_sentences],
        )

        default_text = (
            "The company reported strong profits this quarter."
            if selected_example == "Custom sentence"
            else selected_example
        )

        user_text = st.text_area(
            "Financial sentence",
            default_text,
            height=130,
        )

    with settings_column:
        selected_models = st.multiselect(
            "Live models",
            models,
            default=models,
            format_func=lambda model: DISPLAY_NAMES[model],
        )
        confidence_threshold = st.slider(
            "Confidence threshold",
            min_value=0.50,
            max_value=0.95,
            value=0.70,
            step=0.05,
        )
        st.caption("Lower-confidence predictions are marked for interpretation.")

    if st.button("Analyze all selected models", type="primary"):
        if not user_text.strip():
            st.warning("Please enter a financial sentence.")
        elif not selected_models:
            st.warning("Please select at least one model.")
        else:
            predictions_df = predictions_for_text(selected_models, user_text)

            st.subheader("Live Results")
            show_prediction_cards(predictions_df, confidence_threshold)
            show_agreement_summary(predictions_df)
            show_business_report(predictions_df)

            table_df = predictions_df[["Model", "Label", "Confidence"]].copy()
            table_df["Confidence"] = table_df["Confidence"].round(4)
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.download_button(
                "Download live prediction table",
                table_df.to_csv(index=False),
                file_name="live_predictions.csv",
                mime="text/csv",
            )

    with st.expander("Presentation example set"):
        st.markdown(
            "<span class='small-muted'>Positive, negative, and neutral finance examples.</span>",
            unsafe_allow_html=True,
        )

        if st.button("Run presentation examples", key="run_presentation_examples"):
            if not selected_models:
                st.warning("Please select at least one model.")
            else:
                example_rows = []

                for sentence in example_sentences:
                    for model_name in selected_models:
                        with st.spinner(f"Running {DISPLAY_NAMES[model_name]} on example sentences..."):
                            prediction = predict_sentiment(model_name, sentence)
                            example_rows.append(
                                {
                                    "Sentence": sentence,
                                    "Model": prediction["Model"],
                                    "Label": prediction["Label"],
                                    "Confidence": round(prediction["Confidence"], 4),
                                }
                            )

                example_df = pd.DataFrame(example_rows)
                st.dataframe(example_df, use_container_width=True, hide_index=True)

                label_view = example_df.pivot(
                    index="Sentence",
                    columns="Model",
                    values="Label",
                ).reset_index()
                st.subheader("Compact Label View")
                st.dataframe(label_view, use_container_width=True, hide_index=True)


def show_batch_analyzer(models):
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">Batch finance text analyzer</div>
            <p class="hero-copy">
                Analyze several finance sentences at once and export the output.
                This is closer to how a real analyst or business user would use the system.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_models = st.multiselect(
        "Models for batch analysis",
        models,
        default=models[:2] if len(models) > 1 else models,
        format_func=lambda model: DISPLAY_NAMES[model],
    )
    batch_text = st.text_area(
        "Enter one financial sentence per line",
        "\n".join(BATCH_EXAMPLES),
        height=180,
    )

    sentences = [line.strip() for line in batch_text.splitlines() if line.strip()]
    st.caption(f"{len(sentences)} sentence(s) ready. Keep the batch small for a live demo.")

    if st.button("Analyze Batch", type="primary"):
        if not selected_models:
            st.warning("Please select at least one model.")
            return

        if not sentences:
            st.warning("Please enter at least one sentence.")
            return

        if len(sentences) > 20:
            st.warning("For the live demo, please use 20 sentences or fewer.")
            return

        rows = []
        progress = st.progress(0.0)
        total_runs = len(sentences) * len(selected_models)
        completed_runs = 0

        for sentence in sentences:
            for model_name in selected_models:
                prediction = predict_sentiment(model_name, sentence)
                rows.append(
                    {
                        "Sentence": sentence,
                        "Model": prediction["Model"],
                        "Label": prediction["Label"],
                        "Confidence": round(prediction["Confidence"], 4),
                    }
                )
                completed_runs += 1
                progress.progress(completed_runs / total_runs)

        batch_df = pd.DataFrame(rows)
        st.subheader("Batch Results")
        st.dataframe(batch_df, use_container_width=True, hide_index=True)

        distribution = (
            batch_df.groupby(["Model", "Label"])
            .size()
            .reset_index(name="Count")
        )
        st.subheader("Sentiment Distribution")
        st.dataframe(distribution, use_container_width=True, hide_index=True)

        chart_data = distribution.pivot(
            index="Model",
            columns="Label",
            values="Count",
        ).fillna(0)
        st.bar_chart(chart_data)

        st.download_button(
            "Download batch analysis",
            batch_df.to_csv(index=False),
            file_name="batch_sentiment_analysis.csv",
            mime="text/csv",
        )


st.set_page_config(
    page_title="Financial Sentiment Analysis",
    layout="wide",
)
inject_styles()

st.title(PRODUCT_NAME)
st.write(
    "A prototype decision-support tool for financial text monitoring, "
    "model comparison, and sentiment reporting."
)

models = available_live_models()
final_results = load_csv(MODEL_COMPARISON_PATH)

example_sentences = [
    "The company reported strong revenue growth.",
    "The company announced major losses.",
    "The company will hold its annual meeting next week.",
]

with st.sidebar:
    st.header("Demo Flow")
    st.write("1. Product Dashboard")
    st.write("2. Live Demo")
    st.write("3. Batch Analyzer")
    st.write("4. Final Results")

    st.header("Models")
    st.write("Models used for live comparison:")
    for model in models:
        st.write(f"- {DISPLAY_NAMES[model]}")

    if not saved_model_exists(BERT_LARGE_FINETUNED_DIR):
        st.info(
            "Fine-tuned BERT-large is included in the saved results table, "
            "but live prediction needs the saved model folder."
        )
    else:
        st.info("The first BERT-large prediction may take longer because the model is large.")

tabs = st.tabs(
    [
        "Product Dashboard",
        "Live Demo",
        "Batch Analyzer",
        "Final Comparison",
        "Epoch Experiment",
        "Dataset Size Experiment",
        "Analysis",
        "Prediction Files",
    ]
)

with tabs[0]:
    show_product_dashboard(final_results, models)

with tabs[1]:
    show_live_demo(models, example_sentences)

with tabs[2]:
    show_batch_analyzer(models)

with tabs[3]:
    show_results_overview(final_results)

    show_table(
        "Final Results Table",
        MODEL_COMPARISON_PATH,
        "No final results table found. Run `python 06_results.py` first.",
    )

    show_image_grid(
        "Final Comparison Graphs",
        [
            RESULTS_DIR / "final_accuracy.png",
            RESULTS_DIR / "final_f1.png",
            RESULTS_DIR / "final_time.png",
            RESULTS_DIR / "accuracy_comparison.png",
            RESULTS_DIR / "f1_score_comparison.png",
            RESULTS_DIR / "time_comparison.png",
            RESULTS_DIR / "memory_comparison.png",
        ],
    )

with tabs[4]:
    show_table(
        "Epoch Experiment Table",
        EPOCH_EXPERIMENT_PATH,
        "No epoch experiment table found. Run `python 08_experiment_epochs.py` first.",
    )

    show_image_grid(
        "Epoch Experiment Graphs",
        [
            RESULTS_DIR / "experiment_epochs_scores.png",
            RESULTS_DIR / "experiment_epochs_loss.png",
            RESULTS_DIR / "experiment_epochs_time.png",
            RESULTS_DIR / "experiment_epochs_scores_zoomed.png",
            RESULTS_DIR / "experiment_epochs_time_bar.png",
        ],
    )

with tabs[5]:
    show_table(
        "Dataset Size Experiment Table",
        DATA_SIZE_EXPERIMENT_PATH,
        "No dataset size experiment table found. Run `python 09_experiment_data_size.py` first.",
    )

    show_image_grid(
        "Dataset Size Experiment Graphs",
        [
            RESULTS_DIR / "experiment_data_size_scores.png",
            RESULTS_DIR / "experiment_data_size_time_bar.png",
            RESULTS_DIR / "experiment_data_size_efficiency.png",
            RESULTS_DIR / "experiment_data_size_scores_zoomed.png",
            RESULTS_DIR / "experiment_data_size_time.png",
        ],
    )

with tabs[6]:
    show_markdown_file("Interpretation, Limitations, and Future Work", ANALYSIS_SUMMARY_PATH)

    show_image_grid(
        "Architecture and Main Result Figure",
        [
            RESULTS_DIR / "architecture_diagram.png",
            RESULTS_DIR / "main_summary_figure.png",
        ],
        columns_per_row=1,
    )

    show_image_grid(
        "Confusion Matrices",
        [
            RESULTS_DIR / "confusion_matrix_finbert.png",
            RESULTS_DIR / "confusion_matrix_distilbert_ft.png",
            RESULTS_DIR / "confusion_matrix_bert_large.png",
        ],
    )

with tabs[7]:
    st.write("Preview saved prediction files used in the experiments.")

    show_table(
        "Baseline, BERT Star-Rating, and FinBERT Predictions",
        COMBINED_PREDICTIONS_PATH,
        "No combined predictions file found. Run `python 05_evaluation.py` first.",
        max_rows=20,
    )

    show_table(
        "Fine-tuned DistilBERT Predictions",
        DISTILBERT_PREDICTIONS_PATH,
        "No fine-tuned DistilBERT loaded prediction file found.",
        max_rows=20,
    )

    show_table(
        "Fine-tuned BERT-large Predictions",
        BERT_LARGE_PREDICTIONS_PATH,
        "No fine-tuned BERT-large prediction file found.",
        max_rows=20,
    )
