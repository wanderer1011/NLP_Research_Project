#!/usr/bin/env python3
"""
Distance-based clustering experiment on per-dialogue emotion vectors.

For each dialogue:
- Build an emotion probability vector using
  j-hartmann/emotion-english-distilroberta-base.
- Assign a label:
  - non-manipulative -> none
  - manipulative -> technique label(s) from dataset

Then:
- Cluster vectors in original high-dimensional space.
- Save cluster assignments per dialogue.
- Save cluster summaries and label composition tables.
- Save visual diagnostics (heatmaps + silhouette scan).
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data_loader import Dialogue, load_dataset


def _clean_text(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\bPerson\s*([12])\s*:", r"Person\1:", text)
    return " ".join(text.split())


def _joined_technique_label(dialogue: Dialogue) -> str:
    if dialogue.label == 0:
        return "none"
    if dialogue.techniques:
        return "|".join(dialogue.techniques)
    return "unlabeled_manipulative"


def _plot_label(dialogue: Dialogue) -> str:
    if dialogue.label == 0:
        return "none"
    if dialogue.techniques:
        return dialogue.techniques[0]
    return "unlabeled_manipulative"


def _build_emotion_vectors(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, list[str]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_probs: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1)

        all_probs.append(probs.cpu().numpy().astype(np.float32))

    vectors = np.vstack(all_probs)

    id_to_label = model.config.id2label
    labels = [id_to_label[i] for i in sorted(id_to_label.keys())]
    return vectors, labels


def _cluster_once(
    vectors: np.ndarray,
    method: str,
    n_clusters: int,
    distance_metric: str,
    random_state: int,
) -> np.ndarray:
    if method == "agglomerative":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=distance_metric,
            linkage="average",
        )
        return model.fit_predict(vectors).astype(np.int32)

    # KMeans uses euclidean distance internally.
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    return model.fit_predict(vectors).astype(np.int32)


def _choose_cluster_count(
    vectors: np.ndarray,
    method: str,
    distance_metric: str,
    random_state: int,
    requested_clusters: int,
    min_clusters: int,
    max_clusters: int,
) -> tuple[np.ndarray, int, dict[int, float], float]:
    n_samples = len(vectors)
    if n_samples < 3:
        raise ValueError("Need at least 3 samples for clustering.")

    if requested_clusters > 1:
        labels = _cluster_once(
            vectors=vectors,
            method=method,
            n_clusters=requested_clusters,
            distance_metric=distance_metric,
            random_state=random_state,
        )
        metric_for_sil = "euclidean" if method == "kmeans" else distance_metric
        sil = silhouette_score(vectors, labels, metric=metric_for_sil)
        return labels, requested_clusters, {requested_clusters: float(sil)}, float(sil)

    low = max(2, int(min_clusters))
    high = min(int(max_clusters), max(2, n_samples - 1))
    if low > high:
        low = 2
        high = max(2, n_samples - 1)

    metric_for_sil = "euclidean" if method == "kmeans" else distance_metric

    best_k = low
    best_score = -1e9
    best_labels: np.ndarray | None = None
    score_by_k: dict[int, float] = {}

    for k in range(low, high + 1):
        labels = _cluster_once(
            vectors=vectors,
            method=method,
            n_clusters=k,
            distance_metric=distance_metric,
            random_state=random_state,
        )

        if len(np.unique(labels)) < 2:
            continue

        score = float(silhouette_score(vectors, labels, metric=metric_for_sil))
        score_by_k[k] = score

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_labels is None:
        # Fallback to k=2 if silhouette scan is degenerate.
        best_k = 2
        best_labels = _cluster_once(
            vectors=vectors,
            method=method,
            n_clusters=best_k,
            distance_metric=distance_metric,
            random_state=random_state,
        )
        best_score = float(silhouette_score(vectors, best_labels, metric=metric_for_sil))
        score_by_k[best_k] = best_score

    return best_labels, best_k, score_by_k, best_score


def _save_cluster_assignments(
    output_dir: Path,
    dialogues: list[Dialogue],
    cleaned_texts: list[str],
    emotion_vectors: np.ndarray,
    emotion_names: list[str],
    labels: np.ndarray,
) -> Path:
    out_path = output_dir / "emotion_vectors_with_clusters.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "id",
            "binary_label",
            "technique_label",
            "plot_label",
            "cluster_id",
            "text",
        ] + [f"emotion_{name}" for name in emotion_names]
        writer.writerow(header)

        for i, d in enumerate(dialogues):
            writer.writerow(
                [
                    d.id,
                    d.label,
                    _joined_technique_label(d),
                    _plot_label(d),
                    int(labels[i]),
                    cleaned_texts[i],
                ]
                + [float(v) for v in emotion_vectors[i]]
            )

    return out_path


def _build_cluster_label_matrix(
    labels: np.ndarray,
    plot_labels: list[str],
    top_label_count: int,
) -> tuple[np.ndarray, list[int], list[str]]:
    k_values = sorted(int(v) for v in np.unique(labels))
    label_counts = Counter(plot_labels)

    top_labels = [name for name, _ in label_counts.most_common(max(1, top_label_count))]
    if "other" in top_labels:
        top_labels = [x for x in top_labels if x != "other"]
    col_labels = top_labels + ["other"]

    col_index = {name: i for i, name in enumerate(col_labels)}
    row_index = {k: i for i, k in enumerate(k_values)}

    matrix = np.zeros((len(k_values), len(col_labels)), dtype=np.int32)

    for c_id, lbl in zip(labels.tolist(), plot_labels):
        r = row_index[int(c_id)]
        key = lbl if lbl in top_labels else "other"
        matrix[r, col_index[key]] += 1

    return matrix, k_values, col_labels


def _save_label_matrix_csv(
    output_dir: Path,
    matrix: np.ndarray,
    row_clusters: list[int],
    col_labels: list[str],
) -> Path:
    out_path = output_dir / "cluster_label_matrix.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id"] + col_labels)
        for r, cluster_id in enumerate(row_clusters):
            writer.writerow([cluster_id] + matrix[r].tolist())
    return out_path


def _save_cluster_summary(
    output_dir: Path,
    labels: np.ndarray,
    plot_labels: list[str],
    joined_techniques: list[str],
    vectors: np.ndarray,
    emotion_names: list[str],
) -> Path:
    out_path = output_dir / "cluster_summary.csv"

    unique_clusters = sorted(int(v) for v in np.unique(labels))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "cluster_id",
                "size",
                "top_plot_label",
                "top_plot_label_share",
                "top_joined_technique",
                "top_joined_technique_share",
                "dominant_emotion",
                "dominant_emotion_mean",
            ]
        )

        for c in unique_clusters:
            idx = np.where(labels == c)[0]
            size = len(idx)
            cluster_plot_labels = [plot_labels[i] for i in idx]
            cluster_joined = [joined_techniques[i] for i in idx]

            plot_counter = Counter(cluster_plot_labels)
            top_plot, top_plot_n = plot_counter.most_common(1)[0]

            tech_counter = Counter(cluster_joined)
            top_joined, top_joined_n = tech_counter.most_common(1)[0]

            mean_vec = vectors[idx].mean(axis=0)
            emo_i = int(np.argmax(mean_vec))
            dominant_emo = emotion_names[emo_i]
            dominant_emo_score = float(mean_vec[emo_i])

            writer.writerow(
                [
                    c,
                    size,
                    top_plot,
                    round(top_plot_n / max(size, 1), 4),
                    top_joined,
                    round(top_joined_n / max(size, 1), 4),
                    dominant_emo,
                    round(dominant_emo_score, 4),
                ]
            )

    return out_path


def _plot_heatmap(
    matrix: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    out_path: Path,
    cmap: str,
) -> None:
    fig_w = max(9.0, 0.6 * len(x_labels))
    fig_h = max(5.5, 0.45 * len(y_labels))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(matrix, aspect="auto", cmap=cmap)
    plt.colorbar(shrink=0.85)

    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha="right")
    plt.yticks(range(len(y_labels)), y_labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def _save_cluster_visuals(
    output_dir: Path,
    labels: np.ndarray,
    vectors: np.ndarray,
    emotion_names: list[str],
    label_matrix: np.ndarray,
    row_clusters: list[int],
    col_labels: list[str],
    score_by_k: dict[int, float],
) -> tuple[Path, Path, Path | None]:
    label_heatmap_path = output_dir / "cluster_label_heatmap.png"
    _plot_heatmap(
        matrix=label_matrix,
        x_labels=col_labels,
        y_labels=[f"cluster_{c}" for c in row_clusters],
        title="Cluster vs Technique Label Composition",
        out_path=label_heatmap_path,
        cmap="viridis",
    )

    unique_clusters = sorted(int(v) for v in np.unique(labels))
    mean_matrix = np.zeros((len(unique_clusters), vectors.shape[1]), dtype=np.float32)
    for i, c in enumerate(unique_clusters):
        idx = np.where(labels == c)[0]
        mean_matrix[i] = vectors[idx].mean(axis=0)

    emotion_heatmap_path = output_dir / "cluster_emotion_profile_heatmap.png"
    _plot_heatmap(
        matrix=mean_matrix,
        x_labels=emotion_names,
        y_labels=[f"cluster_{c}" for c in unique_clusters],
        title="Cluster Emotion Profile (Mean Probability)",
        out_path=emotion_heatmap_path,
        cmap="magma",
    )

    silhouette_path: Path | None = None
    if len(score_by_k) > 1:
        silhouette_path = output_dir / "silhouette_by_k.png"
        ks = sorted(score_by_k.keys())
        vals = [score_by_k[k] for k in ks]

        plt.figure(figsize=(8.0, 4.8))
        plt.plot(ks, vals, marker="o", linewidth=1.8)
        plt.xlabel("k (number of clusters)")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette Scan")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(silhouette_path, dpi=250)
        plt.close()

    return label_heatmap_path, emotion_heatmap_path, silhouette_path


def _save_label_counts(output_dir: Path, plot_labels: list[str]) -> Path:
    out_path = output_dir / "label_counts.csv"
    counts = Counter(plot_labels)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["plot_label", "count"])
        for label, n in counts.most_common():
            writer.writerow([label, n])
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster dialogues using distance over emotion vectors."
    )
    parser.add_argument(
        "--dataset",
        default="../iap_pipeline/mentalmanip_con.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--output",
        default="emotion_cluster_experiment",
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        default="j-hartmann/emotion-english-distilroberta-base",
        help="HuggingFace emotion model",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        dest="max_dialogues",
        help="Max number of dialogues to process (0 = all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Max token length for each dialogue",
    )
    parser.add_argument(
        "--cluster-method",
        choices=["agglomerative", "kmeans"],
        default="agglomerative",
        help="Clustering algorithm",
    )
    parser.add_argument(
        "--distance",
        choices=["cosine", "euclidean"],
        default="cosine",
        help="Distance metric (for agglomerative + silhouette metric)",
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=0,
        help="Fixed cluster count; 0 = auto-select via silhouette",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
        help="Min k to scan when auto-selecting clusters",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=12,
        help="Max k to scan when auto-selecting clusters",
    )
    parser.add_argument(
        "--top-label-count",
        type=int,
        default=12,
        help="Top N labels shown explicitly in cluster-label matrix; rest become other",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for kmeans and selection scan",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    dialogues = load_dataset(args.dataset)
    if args.max_dialogues > 0:
        dialogues = dialogues[: args.max_dialogues]
    print(f"Dialogues loaded: {len(dialogues)}")

    cleaned_texts = [_clean_text(d.text) for d in dialogues]
    plot_labels = [_plot_label(d) for d in dialogues]
    joined_techniques = [_joined_technique_label(d) for d in dialogues]

    print(f"Building emotion vectors with model: {args.model}")
    vectors, emotion_names = _build_emotion_vectors(
        texts=cleaned_texts,
        model_name=args.model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("Clustering vectors...")
    cluster_ids, chosen_k, score_by_k, best_score = _choose_cluster_count(
        vectors=vectors,
        method=args.cluster_method,
        distance_metric=args.distance,
        random_state=args.random_state,
        requested_clusters=args.num_clusters,
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
    )

    print(f"Cluster method: {args.cluster_method}")
    print(f"Chosen k: {chosen_k}")
    print(f"Silhouette: {best_score:.4f}")

    assign_path = _save_cluster_assignments(
        output_dir=out_dir,
        dialogues=dialogues,
        cleaned_texts=cleaned_texts,
        emotion_vectors=vectors,
        emotion_names=emotion_names,
        labels=cluster_ids,
    )

    label_count_path = _save_label_counts(out_dir, plot_labels)

    matrix, row_clusters, col_labels = _build_cluster_label_matrix(
        labels=cluster_ids,
        plot_labels=plot_labels,
        top_label_count=args.top_label_count,
    )
    matrix_path = _save_label_matrix_csv(
        output_dir=out_dir,
        matrix=matrix,
        row_clusters=row_clusters,
        col_labels=col_labels,
    )

    summary_path = _save_cluster_summary(
        output_dir=out_dir,
        labels=cluster_ids,
        plot_labels=plot_labels,
        joined_techniques=joined_techniques,
        vectors=vectors,
        emotion_names=emotion_names,
    )

    label_heatmap_path, emotion_heatmap_path, silhouette_path = _save_cluster_visuals(
        output_dir=out_dir,
        labels=cluster_ids,
        vectors=vectors,
        emotion_names=emotion_names,
        label_matrix=matrix,
        row_clusters=row_clusters,
        col_labels=col_labels,
        score_by_k=score_by_k,
    )

    print(f"Saved cluster assignments: {assign_path}")
    print(f"Saved label counts: {label_count_path}")
    print(f"Saved cluster-label matrix: {matrix_path}")
    print(f"Saved cluster summary: {summary_path}")
    print(f"Saved cluster-label heatmap: {label_heatmap_path}")
    print(f"Saved cluster-emotion heatmap: {emotion_heatmap_path}")
    if silhouette_path is not None:
        print(f"Saved silhouette scan: {silhouette_path}")


if __name__ == "__main__":
    main()
