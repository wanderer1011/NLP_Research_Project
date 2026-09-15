"""
Configuration for the Similarity/Retrieval-Based Detection pipeline.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # --- Dataset ---
    # Path to the MentalManip CSV file
    dataset_path: str = "mentalmanip_maj.csv"

    # --- Pipeline ---
    # Max dialogues to process (0 = all)
    max_dialogues: int = 0

    # Sentence Transformer model to use for embeddings
    embedding_model: str = "j-hartmann/emotion-english-distilroberta-base"

    # Number of nearest neighbors to retrieve for binary detection
    k_neighbors: int = 5

    # Optional dedicated k for technique/vulnerability retrieval.
    # If None, falls back to max(k_neighbors, top_matches_to_save)
    k_neighbors_labels: int | None = None

    # Retrieval mode:
    # - "global": single mixed index
    # - "per_class": separate indices for manipulative/non-manipulative and ratio scoring
    # - "linear": logistic regression on embeddings
    # - "auto": evaluate all modes on validation and select the better one
    retrieval_mode: str = "auto"

    # Validation split from train set for threshold tuning
    val_ratio: float = 0.15

    # Optional fixed decision threshold; if None, tune on validation split
    decision_threshold: float | None = None

    # Metric used to tune threshold: accuracy, precision, recall, or f1
    threshold_metric: str = "f1"

    # Minimum recall target used during threshold tuning.
    # Candidates below this recall are rejected when possible.
    min_recall_target: float = 0.90

    # Optional upper bound for false positive rate during threshold tuning.
    # If None, no explicit cap is used.
    max_fpr_target: float | None = None

    # Penalize false positives directly in threshold/mode selection.
    # objective = chosen_metric - fpr_penalty * FPR
    fpr_penalty: float = 0.35

    # If no candidate satisfies max_fpr_target, force fallback to the
    # minimum-FPR threshold before optimizing the objective.
    hard_fpr_fallback: bool = True

    # Number of threshold candidates in [0.01, 0.99] for tuning.
    threshold_steps: int = 99

    # Raises influence of high-similarity neighbors in weighted voting
    similarity_power: float = 2.0

    # Optional score calibration method: "none", "platt", "isotonic"
    calibration_method: str = "none"

    # Mark prediction as uncertain when |score - threshold| <= margin.
    uncertainty_margin: float = 0.0

    # Batch size for embedding generation
    embedding_batch_size: int = 64

    # Normalize embeddings so cosine similarity is stable
    normalize_embeddings: bool = True

    # Number of nearest exemplars to save per test prediction
    top_matches_to_save: int = 3

    # Technique prediction settings (multi-label)
    # Technique scoring mode:
    # - "neighbors": weighted neighbor label voting only
    # - "linear": one-vs-rest logistic regression on embeddings
    # - "hybrid": blend linear probabilities with neighbor votes
    technique_prediction_mode: str = "hybrid"

    # Keep techniques with normalized score >= threshold
    technique_min_score: float = 0.45

    # Maximum number of technique labels to emit per dialogue
    technique_max_labels: int = 3

    # Minimum number of positives required for a technique to be modeled
    # by the linear predictor.
    technique_min_support: int = 5

    # Weight for linear scores in hybrid mode; neighbor vote weight is
    # (1 - technique_linear_weight).
    technique_linear_weight: float = 0.70

    # Force at least one technique label when predicted manipulative.
    technique_force_one: bool = False

    # Exclude manipulative examples without any technique annotation from
    # technique/vulnerability metric denominators.
    exclude_unlabeled_positive_from_label_metrics: bool = True

    # Ratio of training data (0.0 to 1.0)
    train_ratio: float = 0.8

    # Seed for random split
    random_state: int = 42

    # Number of repeated runs with different seeds.
    num_runs: int = 1

    # Seed increment between runs.
    seed_step: int = 101

    # Save threshold sweep points for analysis.
    save_threshold_curve: bool = True

    # Cache embeddings to speed up iterative experiments.
    cache_embeddings: bool = True

    # Directory for cached artifacts.
    cache_dir: str = ".cache"

    # Include raw dialogue text in predictions.csv
    include_text_in_output: bool = True

    # --- Output ---
    output_dir: str = "similarity_results"
