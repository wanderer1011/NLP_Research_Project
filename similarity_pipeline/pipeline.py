import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from config import Config
from data_loader import load_dataset
from evaluation import Metrics, evaluate


def _clean_text(text: str) -> str:
    """Normalize whitespace while preserving dialogue content."""
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\bPerson\s*([12])\s*:", r"Person\1:", text)
    return " ".join(text.split())


def _parse_multi_labels(items: list[str]) -> list[str]:
    """Normalize and split annotation labels into atomic values."""
    labels: list[str] = []
    seen: set[str] = set()
    for item in items:
        for part in re.split(r"[|,]", item or ""):
            label = part.strip()
            if not label:
                continue
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
    return labels


def _build_retriever(train_embeddings: np.ndarray, k_neighbors: int) -> NearestNeighbors:
    k = min(max(1, k_neighbors), len(train_embeddings))
    retriever = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    retriever.fit(train_embeddings)
    return retriever


def _build_class_retrievers(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    k_neighbors: int,
) -> tuple[NearestNeighbors, NearestNeighbors]:
    pos_idx = np.where(train_labels == 1)[0]
    neg_idx = np.where(train_labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Training split must contain both positive and negative classes.")

    pos_retriever = _build_retriever(train_embeddings[pos_idx], k_neighbors)
    neg_retriever = _build_retriever(train_embeddings[neg_idx], k_neighbors)
    return pos_retriever, neg_retriever


def _score_by_neighbors(
    retriever: NearestNeighbors,
    query_embeddings: np.ndarray,
    train_labels: np.ndarray,
    similarity_power: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances, neighbor_idx = retriever.kneighbors(query_embeddings)
    similarities = np.clip(1.0 - distances, 0.0, 1.0)

    neighbor_labels = train_labels[neighbor_idx]
    weights = np.power(similarities + 1e-8, similarity_power)
    denom = np.maximum(weights.sum(axis=1), 1e-8)
    scores = (weights * neighbor_labels).sum(axis=1) / denom
    return scores, similarities, neighbor_idx


def _score_by_per_class_neighbors(
    query_embeddings: np.ndarray,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    k_neighbors: int,
    similarity_power: float,
) -> np.ndarray:
    pos_retriever, neg_retriever = _build_class_retrievers(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        k_neighbors=k_neighbors,
    )

    pos_distances, _ = pos_retriever.kneighbors(query_embeddings)
    neg_distances, _ = neg_retriever.kneighbors(query_embeddings)

    pos_sim = np.clip(1.0 - pos_distances, 0.0, 1.0)
    neg_sim = np.clip(1.0 - neg_distances, 0.0, 1.0)

    pos_weight = np.power(pos_sim + 1e-8, similarity_power).sum(axis=1)
    neg_weight = np.power(neg_sim + 1e-8, similarity_power).sum(axis=1)

    scores = pos_weight / np.maximum(pos_weight + neg_weight, 1e-8)
    return scores


def _score_by_linear_model(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    random_state: int,
) -> np.ndarray:
    model = LogisticRegression(max_iter=2000, random_state=random_state)
    model.fit(train_embeddings, train_labels)
    return model.predict_proba(query_embeddings)[:, 1]


def _compute_scores_for_mode(
    retrieval_mode: str,
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    k_neighbors: int,
    similarity_power: float,
    random_state: int,
) -> np.ndarray:
    if retrieval_mode == "linear":
        return _score_by_linear_model(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            query_embeddings=query_embeddings,
            random_state=random_state,
        )

    if retrieval_mode == "per_class":
        return _score_by_per_class_neighbors(
            query_embeddings=query_embeddings,
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            k_neighbors=k_neighbors,
            similarity_power=similarity_power,
        )

    retriever = _build_retriever(train_embeddings, k_neighbors)
    scores, _, _ = _score_by_neighbors(
        retriever=retriever,
        query_embeddings=query_embeddings,
        train_labels=train_labels,
        similarity_power=similarity_power,
    )
    return scores


def _metric_value(metric_name: str, metrics: Metrics) -> float:
    if metric_name == "accuracy":
        return metrics.accuracy
    if metric_name == "precision":
        return metrics.precision
    if metric_name == "recall":
        return metrics.recall
    return metrics.f1


def _false_positive_rate(metrics: Metrics) -> float:
    denom = metrics.fp + metrics.tn
    return metrics.fp / denom if denom else 0.0


def _objective(metric_name: str, metrics: Metrics, fpr_penalty: float) -> float:
    return _metric_value(metric_name, metrics) - (fpr_penalty * _false_positive_rate(metrics))


def _tune_threshold(
    scores: np.ndarray,
    y_true: list[int],
    metric_name: str,
    min_recall_target: float,
    max_fpr_target: float | None,
    threshold_steps: int,
    fpr_penalty: float,
    hard_fpr_fallback: bool,
) -> tuple[float, Metrics, bool, list[dict[str, float | bool]]]:
    steps = max(5, threshold_steps)
    candidates = np.linspace(0.01, 0.99, steps)

    feasible: list[tuple[float, Metrics]] = []
    fallback: list[tuple[float, Metrics]] = []
    curve: list[dict[str, float | bool]] = []

    for threshold in candidates:
        preds = [1 if s >= threshold else 0 for s in scores]
        metrics = evaluate(y_true, preds)
        fpr = _false_positive_rate(metrics)
        recall_ok = metrics.recall >= min_recall_target
        fpr_ok = True if max_fpr_target is None else fpr <= max_fpr_target
        is_feasible = recall_ok and fpr_ok

        curve.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "fpr": fpr,
                "feasible": is_feasible,
            }
        )

        fallback.append((float(threshold), metrics))
        if is_feasible:
            feasible.append((float(threshold), metrics))

    candidates_to_rank = feasible if feasible else fallback

    if not feasible and hard_fpr_fallback and max_fpr_target is not None:
        min_fpr = min(_false_positive_rate(m) for _, m in fallback)
        candidates_to_rank = [
            (t, m)
            for t, m in fallback
            if _false_positive_rate(m) == min_fpr
        ]

    best_threshold, best_metrics = candidates_to_rank[0]
    best_obj = _objective(metric_name, best_metrics, fpr_penalty)
    best_fpr = _false_positive_rate(best_metrics)

    for threshold, metrics in candidates_to_rank[1:]:
        obj = _objective(metric_name, metrics, fpr_penalty)
        fpr = _false_positive_rate(metrics)

        if (
            obj > best_obj
            or (obj == best_obj and fpr < best_fpr)
            or (obj == best_obj and fpr == best_fpr and metrics.recall > best_metrics.recall)
        ):
            best_threshold = threshold
            best_metrics = metrics
            best_obj = obj
            best_fpr = fpr

    constraints_satisfied = len(feasible) > 0
    return best_threshold, best_metrics, constraints_satisfied, curve


def _fit_calibrator(
    method: str,
    scores: np.ndarray,
    y_true: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray] | None:
    if method == "none":
        return None

    unique = np.unique(y_true)
    if len(unique) < 2:
        return None

    if method == "platt":
        model = LogisticRegression(max_iter=2000)
        model.fit(scores.reshape(-1, 1), y_true)

        def _apply(values: np.ndarray) -> np.ndarray:
            return model.predict_proba(values.reshape(-1, 1))[:, 1]

        return _apply

    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(scores, y_true)

        def _apply(values: np.ndarray) -> np.ndarray:
            return np.asarray(model.predict(values), dtype=np.float32)

        return _apply

    return None


def _apply_calibrator(
    calibrator: Callable[[np.ndarray], np.ndarray] | None,
    values: np.ndarray,
) -> np.ndarray:
    if calibrator is None:
        return values
    return np.clip(np.asarray(calibrator(values), dtype=np.float32), 0.0, 1.0)


def _rank_labels_from_neighbors(
    neighbor_global_idx: list[int],
    neighbor_similarities: np.ndarray,
    binary_labels: np.ndarray,
    labels_by_dialogue: list[list[str]],
    similarity_power: float,
) -> list[tuple[str, float]]:
    """Aggregate weighted label scores from manipulative neighbors only."""
    scores: dict[str, float] = defaultdict(float)
    display_name: dict[str, str] = {}
    total_weight = 0.0

    for neighbor_idx, sim in zip(neighbor_global_idx, neighbor_similarities):
        if int(binary_labels[neighbor_idx]) != 1:
            continue

        weight = float(np.power(float(sim) + 1e-8, similarity_power))
        if weight <= 0.0:
            continue

        labels = labels_by_dialogue[neighbor_idx]
        if not labels:
            continue

        total_weight += weight
        for label in labels:
            key = label.casefold()
            display_name.setdefault(key, label)
            scores[key] += weight

    if total_weight <= 0.0:
        return []

    ranked = [
        (display_name[key], value / total_weight)
        for key, value in scores.items()
    ]
    ranked.sort(key=lambda x: (-x[1], x[0].casefold()))
    return ranked


def _fit_linear_multilabel_predictor(
    train_embeddings: np.ndarray,
    train_binary_labels: np.ndarray,
    labels_by_dialogue: list[list[str]],
    min_support: int,
    random_state: int,
) -> tuple[OneVsRestClassifier, list[str], dict[str, str]] | None:
    """Train a one-vs-rest logistic model for multi-label technique prediction."""
    positive_idx = np.where(train_binary_labels == 1)[0]
    if len(positive_idx) < 8:
        return None

    label_counts: dict[str, int] = defaultdict(int)
    display_name: dict[str, str] = {}

    for idx in positive_idx:
        seen: set[str] = set()
        for label in labels_by_dialogue[int(idx)]:
            key = label.casefold().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            display_name.setdefault(key, label)
            label_counts[key] += 1

    support_floor = max(1, int(min_support))
    class_keys = sorted(key for key, count in label_counts.items() if count >= support_floor)
    if not class_keys:
        return None

    class_to_col = {key: i for i, key in enumerate(class_keys)}

    row_indices: list[int] = []
    targets: list[list[int]] = []

    for idx in positive_idx:
        row = [0] * len(class_keys)
        has_label = False
        seen: set[str] = set()
        for label in labels_by_dialogue[int(idx)]:
            key = label.casefold().strip()
            if key in seen or key not in class_to_col:
                continue
            seen.add(key)
            row[class_to_col[key]] = 1
            has_label = True
        if has_label:
            row_indices.append(int(idx))
            targets.append(row)

    if len(row_indices) < 8:
        return None

    X = train_embeddings[np.array(row_indices, dtype=np.int32)]
    y = np.asarray(targets, dtype=np.int32)

    valid_cols = [j for j in range(y.shape[1]) if 0 < int(y[:, j].sum()) < y.shape[0]]
    if not valid_cols:
        return None

    if len(valid_cols) != y.shape[1]:
        y = y[:, valid_cols]
        class_keys = [class_keys[j] for j in valid_cols]

    classifier = OneVsRestClassifier(
        LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        )
    )
    classifier.fit(X, y)

    display_map = {key: display_name.get(key, key) for key in class_keys}
    return classifier, class_keys, display_map


def _predict_ranked_labels_from_linear(
    classifier: OneVsRestClassifier,
    class_keys: list[str],
    display_map: dict[str, str],
    query_embeddings: np.ndarray,
) -> list[list[tuple[str, float]]]:
    scores = np.asarray(classifier.predict_proba(query_embeddings), dtype=np.float32)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)

    rankings: list[list[tuple[str, float]]] = []
    for row in scores:
        ranked = [
            (display_map.get(class_keys[j], class_keys[j]), float(np.clip(row[j], 0.0, 1.0)))
            for j in range(min(len(class_keys), len(row)))
        ]
        ranked.sort(key=lambda x: (-x[1], x[0].casefold()))
        rankings.append(ranked)
    return rankings


def _blend_ranked_labels(
    primary: list[tuple[str, float]],
    secondary: list[tuple[str, float]],
    primary_weight: float,
) -> list[tuple[str, float]]:
    w_primary = float(np.clip(primary_weight, 0.0, 1.0))
    w_secondary = 1.0 - w_primary

    scores: dict[str, float] = defaultdict(float)
    display_name: dict[str, str] = {}

    for label, score in primary:
        key = label.casefold()
        display_name.setdefault(key, label)
        scores[key] += w_primary * float(score)

    for label, score in secondary:
        key = label.casefold()
        display_name.setdefault(key, label)
        scores[key] += w_secondary * float(score)

    ranked = [(display_name[key], value) for key, value in scores.items()]
    ranked.sort(key=lambda x: (-x[1], x[0].casefold()))
    return ranked


def _select_labels(
    ranked: list[tuple[str, float]],
    min_score: float,
    max_labels: int,
    force_one: bool,
) -> list[str]:
    if not ranked:
        return []

    max_n = max(1, max_labels)
    selected = [label for label, score in ranked if score >= min_score][:max_n]
    if selected:
        return selected

    if force_one:
        return [ranked[0][0]]
    return []


def _compute_multilabel_metrics(
    y_true_binary: list[int],
    true_sets: list[set[str]],
    pred_sets: list[set[str]],
    exclude_unlabeled_positive: bool,
) -> dict[str, float]:
    tp = fp = fn = 0
    exact_match = 0
    jaccard_sum = 0.0
    support = 0

    for y_true, true_set, pred_set in zip(y_true_binary, true_sets, pred_sets):
        if y_true != 1:
            continue
        if exclude_unlabeled_positive and len(true_set) == 0:
            continue

        support += 1
        inter = true_set & pred_set
        union = true_set | pred_set
        tp += len(inter)
        fp += len(pred_set - true_set)
        fn += len(true_set - pred_set)

        if true_set == pred_set:
            exact_match += 1
        jaccard_sum += (len(inter) / len(union)) if union else 1.0

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "support": float(support),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "exact_match": (exact_match / support) if support else 0.0,
        "mean_jaccard": (jaccard_sum / support) if support else 0.0,
    }


def _embedding_cache_path(
    config: Config,
    texts: list[str],
) -> Path:
    hasher = hashlib.sha256()
    hasher.update(config.embedding_model.encode("utf-8"))
    hasher.update(str(config.normalize_embeddings).encode("utf-8"))
    hasher.update(str(len(texts)).encode("utf-8"))
    for text in texts:
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\0")
    key = hasher.hexdigest()[:20]

    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"embeddings_{key}.npy"


def _get_embeddings(
    config: Config,
    model: SentenceTransformer,
    texts: list[str],
) -> np.ndarray:
    if config.cache_embeddings:
        cache_path = _embedding_cache_path(config, texts)
        if cache_path.exists():
            loaded = np.load(cache_path)
            if loaded.shape[0] == len(texts):
                return loaded.astype(np.float32)

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        batch_size=config.embedding_batch_size,
        normalize_embeddings=config.normalize_embeddings,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if config.cache_embeddings:
        cache_path = _embedding_cache_path(config, texts)
        np.save(cache_path, embeddings)

    return embeddings


def _safe_pr_auc(y_true: list[int], scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(y_true, scores))
    except Exception:
        return 0.0


def _safe_roc_auc(y_true: list[int], scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return 0.0


def _run_single_split(
    config: Config,
    run_seed: int,
    run_index: int,
    embeddings: np.ndarray,
    texts: list[str],
    labels: np.ndarray,
    ids: list[str],
    techniques_by_dialogue: list[list[str]],
    vulnerabilities_by_dialogue: list[list[str]],
    output_dir: Path,
) -> tuple[Metrics, dict[str, float | int | str | bool | None]]:
    all_idx = np.arange(len(texts))

    train_idx, test_idx = train_test_split(
        all_idx,
        train_size=config.train_ratio,
        random_state=run_seed,
        stratify=labels,
    )

    selected_mode = "global" if config.retrieval_mode == "auto" else config.retrieval_mode
    selected_calibrator: Callable[[np.ndarray], np.ndarray] | None = None
    decision_threshold = 0.5
    selected_threshold_curve: list[dict[str, float | bool]] = []
    constraints_satisfied = True

    if config.decision_threshold is not None:
        decision_threshold = float(config.decision_threshold)
    elif config.val_ratio > 0.0 and len(train_idx) >= 20:
        index_idx, val_idx = train_test_split(
            train_idx,
            test_size=config.val_ratio,
            random_state=run_seed,
            stratify=labels[train_idx],
        )

        candidate_modes = [selected_mode]
        if config.retrieval_mode == "auto":
            candidate_modes = ["global", "per_class", "linear"]

        best_mode = candidate_modes[0]
        best_threshold = 0.5
        best_metrics = evaluate(labels[val_idx].tolist(), [1] * len(val_idx))
        best_curve: list[dict[str, float | bool]] = []
        best_calibrator: Callable[[np.ndarray], np.ndarray] | None = None
        best_objective = -1e9
        best_pr_auc = -1.0
        best_constraints = False

        for mode in candidate_modes:
            raw_val_scores = _compute_scores_for_mode(
                retrieval_mode=mode,
                train_embeddings=embeddings[index_idx],
                train_labels=labels[index_idx],
                query_embeddings=embeddings[val_idx],
                k_neighbors=config.k_neighbors,
                similarity_power=config.similarity_power,
                random_state=run_seed,
            )

            calibrator = _fit_calibrator(
                method=config.calibration_method,
                scores=raw_val_scores,
                y_true=labels[val_idx],
            )
            val_scores = _apply_calibrator(calibrator, raw_val_scores)

            threshold, val_metrics, mode_constraints, curve = _tune_threshold(
                scores=val_scores,
                y_true=labels[val_idx].tolist(),
                metric_name=config.threshold_metric,
                min_recall_target=config.min_recall_target,
                max_fpr_target=config.max_fpr_target,
                threshold_steps=config.threshold_steps,
                fpr_penalty=config.fpr_penalty,
                hard_fpr_fallback=config.hard_fpr_fallback,
            )

            objective = _objective(config.threshold_metric, val_metrics, config.fpr_penalty)
            mode_pr_auc = _safe_pr_auc(labels[val_idx].tolist(), val_scores)
            mode_fpr = _false_positive_rate(val_metrics)
            best_fpr = _false_positive_rate(best_metrics)

            if (
                objective > best_objective
                or (objective == best_objective and mode_fpr < best_fpr)
                or (objective == best_objective and mode_fpr == best_fpr and mode_pr_auc > best_pr_auc)
            ):
                best_mode = mode
                best_threshold = threshold
                best_metrics = val_metrics
                best_curve = curve
                best_calibrator = calibrator
                best_objective = objective
                best_pr_auc = mode_pr_auc
                best_constraints = mode_constraints

        selected_mode = best_mode
        decision_threshold = best_threshold
        selected_threshold_curve = best_curve
        selected_calibrator = best_calibrator
        constraints_satisfied = best_constraints

        print(
            f"Run {run_index}: tuning mode={selected_mode}, "
            f"threshold={decision_threshold:.3f}, "
            f"{config.threshold_metric}={_metric_value(config.threshold_metric, best_metrics):.4f}, "
            f"fpr={_false_positive_rate(best_metrics):.4f}, "
            f"objective={best_objective:.4f}"
        )
        if not constraints_satisfied:
            print("  Note: constraints unmet in tuning; using safest fallback policy.")
    else:
        if config.retrieval_mode == "auto":
            selected_mode = "global"
        if config.decision_threshold is not None:
            decision_threshold = float(config.decision_threshold)
        else:
            decision_threshold = 0.5

    test_scores_raw = _compute_scores_for_mode(
        retrieval_mode=selected_mode,
        train_embeddings=embeddings[train_idx],
        train_labels=labels[train_idx],
        query_embeddings=embeddings[test_idx],
        k_neighbors=config.k_neighbors,
        similarity_power=config.similarity_power,
        random_state=run_seed,
    )
    test_scores = _apply_calibrator(selected_calibrator, test_scores_raw)

    y_test = labels[test_idx].tolist()
    y_pred = [1 if score >= decision_threshold else 0 for score in test_scores]

    active_technique_mode = config.technique_prediction_mode
    linear_ranked_techniques_all: list[list[tuple[str, float]]] | None = None

    if config.technique_prediction_mode in {"linear", "hybrid"}:
        train_techniques = [techniques_by_dialogue[int(idx)] for idx in train_idx]
        linear_model_bundle = _fit_linear_multilabel_predictor(
            train_embeddings=embeddings[train_idx],
            train_binary_labels=labels[train_idx],
            labels_by_dialogue=train_techniques,
            min_support=config.technique_min_support,
            random_state=run_seed,
        )

        if linear_model_bundle is None:
            active_technique_mode = "neighbors"
            print("  Technique mode fallback: neighbors (insufficient labeled support for linear predictor)")
        else:
            classifier, class_keys, display_map = linear_model_bundle
            linear_ranked_techniques_all = _predict_ranked_labels_from_linear(
                classifier=classifier,
                class_keys=class_keys,
                display_map=display_map,
                query_embeddings=embeddings[test_idx],
            )
            active_technique_mode = config.technique_prediction_mode

    k_labels = config.k_neighbors_labels if config.k_neighbors_labels is not None else max(config.k_neighbors, config.top_matches_to_save)
    report_k = max(max(1, k_labels), max(1, config.top_matches_to_save))
    report_retriever = _build_retriever(embeddings[train_idx], report_k)
    _, test_similarities, test_neighbor_idx = _score_by_neighbors(
        retriever=report_retriever,
        query_embeddings=embeddings[test_idx],
        train_labels=labels[train_idx],
        similarity_power=config.similarity_power,
    )

    top_n = max(1, config.top_matches_to_save)
    per_example_rows: list[dict[str, object]] = []

    true_technique_sets: list[set[str]] = []
    pred_technique_sets: list[set[str]] = []
    true_vulnerability_sets: list[set[str]] = []
    pred_vulnerability_sets: list[set[str]] = []

    for i, global_test_idx in enumerate(test_idx):
        all_neighbor_local_idx = test_neighbor_idx[i]
        all_neighbor_global_idx = [int(train_idx[j]) for j in all_neighbor_local_idx]
        all_neighbor_sims = test_similarities[i]

        ranked_techniques_neighbors = _rank_labels_from_neighbors(
            neighbor_global_idx=all_neighbor_global_idx,
            neighbor_similarities=all_neighbor_sims,
            binary_labels=labels,
            labels_by_dialogue=techniques_by_dialogue,
            similarity_power=config.similarity_power,
        )

        ranked_techniques = ranked_techniques_neighbors
        if linear_ranked_techniques_all is not None:
            ranked_techniques_linear = linear_ranked_techniques_all[i]
            if active_technique_mode == "linear":
                ranked_techniques = ranked_techniques_linear
            elif active_technique_mode == "hybrid":
                ranked_techniques = _blend_ranked_labels(
                    primary=ranked_techniques_linear,
                    secondary=ranked_techniques_neighbors,
                    primary_weight=config.technique_linear_weight,
                )

        ranked_vulnerabilities = _rank_labels_from_neighbors(
            neighbor_global_idx=all_neighbor_global_idx,
            neighbor_similarities=all_neighbor_sims,
            binary_labels=labels,
            labels_by_dialogue=vulnerabilities_by_dialogue,
            similarity_power=config.similarity_power,
        )

        predicted_techniques = _select_labels(
            ranked=ranked_techniques,
            min_score=config.technique_min_score,
            max_labels=config.technique_max_labels,
            force_one=(config.technique_force_one and y_pred[i] == 1),
        )
        predicted_vulnerabilities = _select_labels(
            ranked=ranked_vulnerabilities,
            min_score=config.technique_min_score,
            max_labels=config.technique_max_labels,
            force_one=False,
        )
        if y_pred[i] == 0:
            predicted_techniques = []
            predicted_vulnerabilities = []

        ground_truth_techniques = techniques_by_dialogue[global_test_idx]
        ground_truth_vulnerabilities = vulnerabilities_by_dialogue[global_test_idx]

        true_technique_sets.append(set(t.casefold() for t in ground_truth_techniques))
        pred_technique_sets.append(set(t.casefold() for t in predicted_techniques))
        true_vulnerability_sets.append(set(v.casefold() for v in ground_truth_vulnerabilities))
        pred_vulnerability_sets.append(set(v.casefold() for v in predicted_vulnerabilities))

        top_neighbor_local_idx = all_neighbor_local_idx[:top_n]
        top_neighbor_global_idx = [int(train_idx[j]) for j in top_neighbor_local_idx]

        neighbor_ids = [ids[j] for j in top_neighbor_global_idx]
        neighbor_labels = [int(labels[j]) for j in top_neighbor_global_idx]
        neighbor_sims = [round(float(s), 4) for s in all_neighbor_sims[:top_n]]
        neighbor_techniques = ["|".join(techniques_by_dialogue[j]) for j in top_neighbor_global_idx]
        neighbor_vulnerabilities = ["|".join(vulnerabilities_by_dialogue[j]) for j in top_neighbor_global_idx]

        top_ranked_tech = ranked_techniques[: max(1, config.technique_max_labels)]
        tech_score_text = " | ".join(f"{label}:{score:.3f}" for label, score in top_ranked_tech)

        top_ranked_vuln = ranked_vulnerabilities[: max(1, config.technique_max_labels)]
        vuln_score_text = " | ".join(f"{label}:{score:.3f}" for label, score in top_ranked_vuln)

        top_similarity = float(all_neighbor_sims[0]) if len(all_neighbor_sims) > 0 else 0.0
        second_similarity = float(all_neighbor_sims[1]) if len(all_neighbor_sims) > 1 else top_similarity
        similarity_margin = top_similarity - second_similarity

        top_pos = 0.0
        top_neg = 0.0
        for sim, ng_idx in zip(all_neighbor_sims, all_neighbor_global_idx):
            if int(labels[ng_idx]) == 1:
                top_pos = max(top_pos, float(sim))
            else:
                top_neg = max(top_neg, float(sim))
        class_similarity_margin = top_pos - top_neg

        is_uncertain = abs(float(test_scores[i]) - float(decision_threshold)) <= max(0.0, config.uncertainty_margin)

        row: dict[str, object] = {
            "id": ids[global_test_idx],
            "ground_truth": int(labels[global_test_idx]),
            "predicted": int(y_pred[i]),
            "score": round(float(test_scores[i]), 4),
            "threshold": round(float(decision_threshold), 4),
            "is_uncertain": int(is_uncertain),
            "top_similarity": round(top_similarity, 4),
            "second_similarity": round(second_similarity, 4),
            "similarity_margin": round(similarity_margin, 4),
            "class_similarity_margin": round(class_similarity_margin, 4),
            "ground_truth_techniques": "|".join(ground_truth_techniques),
            "predicted_techniques": "|".join(predicted_techniques),
            "predicted_technique_scores": tech_score_text,
            "ground_truth_vulnerabilities": "|".join(ground_truth_vulnerabilities),
            "predicted_vulnerabilities": "|".join(predicted_vulnerabilities),
            "predicted_vulnerability_scores": vuln_score_text,
            "neighbor_ids": " || ".join(neighbor_ids),
            "neighbor_labels": " || ".join(str(x) for x in neighbor_labels),
            "neighbor_similarities": " || ".join(str(x) for x in neighbor_sims),
            "neighbor_techniques": " || ".join(neighbor_techniques),
            "neighbor_vulnerabilities": " || ".join(neighbor_vulnerabilities),
        }
        if config.include_text_in_output:
            row["text"] = texts[global_test_idx]

        per_example_rows.append(row)

    technique_metrics = _compute_multilabel_metrics(
        y_true_binary=y_test,
        true_sets=true_technique_sets,
        pred_sets=pred_technique_sets,
        exclude_unlabeled_positive=config.exclude_unlabeled_positive_from_label_metrics,
    )
    vulnerability_metrics = _compute_multilabel_metrics(
        y_true_binary=y_test,
        true_sets=true_vulnerability_sets,
        pred_sets=pred_vulnerability_sets,
        exclude_unlabeled_positive=config.exclude_unlabeled_positive_from_label_metrics,
    )

    pr_auc = _safe_pr_auc(y_test, test_scores)
    roc_auc = _safe_roc_auc(y_test, test_scores)

    metrics = evaluate(y_test, y_pred)

    print("\n" + "=" * 50)
    print(f"  Similarity/Retrieval Pipeline - Run {run_index}")
    print("=" * 50)
    print(f"  Seed            : {run_seed}")
    print(f"  Train/Test      : {len(train_idx)}/{len(test_idx)}")
    print(f"  Total test      : {metrics.total}")
    print(f"  Accuracy        : {metrics.accuracy:.4f}")
    print(f"  Precision       : {metrics.precision:.4f}")
    print(f"  Recall          : {metrics.recall:.4f}")
    print(f"  F1 Score        : {metrics.f1:.4f}")
    print(f"  FN Rate         : {metrics.false_negative_rate:.4f}")
    print(f"  FPR             : {_false_positive_rate(metrics):.4f}")
    print(f"  PR-AUC          : {pr_auc:.4f}")
    print(f"  ROC-AUC         : {roc_auc:.4f}")
    print(f"  Active mode     : {selected_mode}")
    print(f"  Calibration     : {config.calibration_method}")
    print(f"  Technique mode  : {active_technique_mode} (requested: {config.technique_prediction_mode})")
    print(f"  Threshold       : {decision_threshold:.3f}")
    print(f"  Constraints met : {constraints_satisfied}")
    print(f"  TP={metrics.tp}  TN={metrics.tn}  FP={metrics.fp}  FN={metrics.fn}")
    print("  Technique labels (manipulative only):")
    print(f"    Micro Precision : {technique_metrics['micro_precision']:.4f}")
    print(f"    Micro Recall    : {technique_metrics['micro_recall']:.4f}")
    print(f"    Micro F1        : {technique_metrics['micro_f1']:.4f}")
    print(f"    Exact Match     : {technique_metrics['exact_match']:.4f}")
    print(f"    Mean Jaccard    : {technique_metrics['mean_jaccard']:.4f}")
    print("  Vulnerability labels (manipulative only):")
    print(f"    Micro Precision : {vulnerability_metrics['micro_precision']:.4f}")
    print(f"    Micro Recall    : {vulnerability_metrics['micro_recall']:.4f}")
    print(f"    Micro F1        : {vulnerability_metrics['micro_f1']:.4f}")
    print(f"    Exact Match     : {vulnerability_metrics['exact_match']:.4f}")
    print(f"    Mean Jaccard    : {vulnerability_metrics['mean_jaccard']:.4f}")
    print("=" * 50)

    output_dir.mkdir(parents=True, exist_ok=True)

    pred_path = output_dir / "predictions.csv"

    base_columns = ["id"]
    if config.include_text_in_output:
        base_columns.append("text")

    columns = base_columns + [
        "ground_truth",
        "predicted",
        "score",
        "threshold",
        "is_uncertain",
        "top_similarity",
        "second_similarity",
        "similarity_margin",
        "class_similarity_margin",
        "ground_truth_techniques",
        "predicted_techniques",
        "predicted_technique_scores",
        "ground_truth_vulnerabilities",
        "predicted_vulnerabilities",
        "predicted_vulnerability_scores",
        "neighbor_ids",
        "neighbor_labels",
        "neighbor_similarities",
        "neighbor_techniques",
        "neighbor_vulnerabilities",
    ]

    with open(pred_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in per_example_rows:
            writer.writerow([row.get(col, "") for col in columns])

    metrics_payload: dict[str, float | int | str | bool | None] = {
        "run_index": run_index,
        "seed": run_seed,
        "total": metrics.total,
        "accuracy": round(metrics.accuracy, 4),
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "f1": round(metrics.f1, 4),
        "false_negative_rate": round(metrics.false_negative_rate, 4),
        "false_positive_rate": round(_false_positive_rate(metrics), 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold": round(float(decision_threshold), 4),
        "threshold_metric": config.threshold_metric,
        "requested_retrieval_mode": config.retrieval_mode,
        "active_retrieval_mode": selected_mode,
        "calibration_method": config.calibration_method,
        "min_recall_target": config.min_recall_target,
        "max_fpr_target": config.max_fpr_target,
        "fpr_penalty": config.fpr_penalty,
        "hard_fpr_fallback": config.hard_fpr_fallback,
        "tuning_constraints_satisfied": constraints_satisfied,
        "k_neighbors": config.k_neighbors,
        "k_neighbors_labels": k_labels,
        "similarity_power": config.similarity_power,
        "uncertainty_margin": config.uncertainty_margin,
        "requested_technique_prediction_mode": config.technique_prediction_mode,
        "active_technique_prediction_mode": active_technique_mode,
        "technique_min_score": config.technique_min_score,
        "technique_max_labels": config.technique_max_labels,
        "technique_min_support": config.technique_min_support,
        "technique_linear_weight": config.technique_linear_weight,
        "technique_force_one": config.technique_force_one,
        "exclude_unlabeled_positive_from_label_metrics": config.exclude_unlabeled_positive_from_label_metrics,
        "technique_support": int(technique_metrics["support"]),
        "technique_micro_precision": round(technique_metrics["micro_precision"], 4),
        "technique_micro_recall": round(technique_metrics["micro_recall"], 4),
        "technique_micro_f1": round(technique_metrics["micro_f1"], 4),
        "technique_exact_match": round(technique_metrics["exact_match"], 4),
        "technique_mean_jaccard": round(technique_metrics["mean_jaccard"], 4),
        "vulnerability_support": int(vulnerability_metrics["support"]),
        "vulnerability_micro_precision": round(vulnerability_metrics["micro_precision"], 4),
        "vulnerability_micro_recall": round(vulnerability_metrics["micro_recall"], 4),
        "vulnerability_micro_f1": round(vulnerability_metrics["micro_f1"], 4),
        "vulnerability_exact_match": round(vulnerability_metrics["exact_match"], 4),
        "vulnerability_mean_jaccard": round(vulnerability_metrics["mean_jaccard"], 4),
        "tp": metrics.tp,
        "tn": metrics.tn,
        "fp": metrics.fp,
        "fn": metrics.fn,
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    if config.save_threshold_curve and selected_threshold_curve:
        threshold_curve_payload = {
            "run_index": run_index,
            "seed": run_seed,
            "mode": selected_mode,
            "threshold_metric": config.threshold_metric,
            "fpr_penalty": config.fpr_penalty,
            "max_fpr_target": config.max_fpr_target,
            "min_recall_target": config.min_recall_target,
            "curve": selected_threshold_curve,
        }
        with open(output_dir / "threshold_curve.json", "w", encoding="utf-8") as f:
            json.dump(threshold_curve_payload, f, indent=2)

    print(f"Saved to {output_dir}")

    return metrics, metrics_payload


def _aggregate_runs(
    output_root: Path,
    per_run_metrics: list[dict[str, float | int | str | bool | None]],
) -> None:
    numeric_keys = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "false_negative_rate",
        "false_positive_rate",
        "pr_auc",
        "roc_auc",
        "threshold",
        "technique_micro_precision",
        "technique_micro_recall",
        "technique_micro_f1",
        "technique_exact_match",
        "technique_mean_jaccard",
        "vulnerability_micro_precision",
        "vulnerability_micro_recall",
        "vulnerability_micro_f1",
        "vulnerability_exact_match",
        "vulnerability_mean_jaccard",
    ]

    aggregate: dict[str, object] = {
        "num_runs": len(per_run_metrics),
        "runs": per_run_metrics,
        "mean": {},
        "std": {},
    }

    for key in numeric_keys:
        values = [float(m[key]) for m in per_run_metrics if key in m and m[key] is not None]
        if not values:
            continue
        aggregate["mean"][key] = round(float(np.mean(values)), 4)
        aggregate["std"][key] = round(float(np.std(values)), 4)

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "aggregate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)


def run_pipeline(config: Config) -> Metrics:
    print(f"Loading dataset from {config.dataset_path} ...")
    dialogues = load_dataset(config.dataset_path)
    if config.max_dialogues > 0:
        dialogues = dialogues[: config.max_dialogues]
    print(f"  -> {len(dialogues)} dialogues loaded.\n")

    print(f"Loading SentenceTransformer model: {config.embedding_model} ...")
    model = SentenceTransformer(config.embedding_model)

    print("Preparing texts and labels...")
    texts = [_clean_text(d.text) for d in dialogues]
    labels = np.array([d.label for d in dialogues], dtype=np.int32)
    ids = [d.id for d in dialogues]
    techniques_by_dialogue = [_parse_multi_labels(d.techniques) for d in dialogues]
    vulnerabilities_by_dialogue = [_parse_multi_labels(d.vulnerabilities) for d in dialogues]

    print("Encoding all dialogues...")
    embeddings = _get_embeddings(config=config, model=model, texts=texts)

    output_root = Path(config.output_dir)
    num_runs = max(1, int(config.num_runs))

    per_run_metrics: list[dict[str, float | int | str | bool | None]] = []
    last_metrics: Metrics | None = None

    for run_idx in range(1, num_runs + 1):
        run_seed = int(config.random_state + (run_idx - 1) * config.seed_step)
        run_output = output_root if num_runs == 1 else output_root / f"run_{run_idx:02d}"

        metrics, run_payload = _run_single_split(
            config=config,
            run_seed=run_seed,
            run_index=run_idx,
            embeddings=embeddings,
            texts=texts,
            labels=labels,
            ids=ids,
            techniques_by_dialogue=techniques_by_dialogue,
            vulnerabilities_by_dialogue=vulnerabilities_by_dialogue,
            output_dir=run_output,
        )
        per_run_metrics.append(run_payload)
        last_metrics = metrics

    if num_runs > 1:
        _aggregate_runs(output_root, per_run_metrics)
        print("\nAggregate summary saved to aggregate_metrics.json")

    assert last_metrics is not None
    return last_metrics
