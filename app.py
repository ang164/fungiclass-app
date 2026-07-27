"""
FungiClass: Streamlit interface for the hierarchical fungal ITS classifier.

Run from the thesis repository root with:
    streamlit run app/app.py

In the standalone deployment repository, where this file is at the root, use:
    streamlit run app.py
"""

from datetime import datetime
from pathlib import Path
import hashlib
import json
import re

import joblib
import numpy as np
import pandas as pd
from Bio import Align
from sklearn.metrics.pairwise import cosine_distances
import streamlit as st


#=============================================================================
# CONFIGURATION
#=============================================================================
st.set_page_config(
    page_title="FungiClass — ITS Identifier",
    page_icon="🍄",
    layout="wide",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        background: #0d0d0d;
        border-right: 1px solid #222222;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 20rem !important;
        max-width: 20rem !important;
    }
    section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 20rem !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0;
    }
    .sidebar-model-info {
        font-size: 0.85rem;
        line-height: 1.9;
    }
    .sidebar-model-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.35rem;
    }
    .info-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 5.75rem;
        width: 5.75rem;
        height: 2.1rem;
        padding: 0.3rem 0.45rem;
        box-sizing: border-box;
        border-radius: 0.25rem;
        background: #1432db;
        color: #ffffff;
        font-size: 0.82rem;
        font-weight: 600;
        line-height: 1;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = Path("models_hier")
LEVELS = ["phylum", "class", "order", "family", "genus", "species"]
TRAINING_MIN_LENGTH = 200
TRAINING_MAX_LENGTH = 1000
EVALUABLE_SPECIES_COUNT = 8_848

DEFAULT_THRESHOLDS = {
    "phylum": 0.84,
    "class": 0.68,
    "order": 0.72,
    "family": 0.64,
    "genus": 0.84,
    "species": 0.84,
}


#=============================================================================
# MODEL LOADING
#=============================================================================
@st.cache_resource(show_spinner="Loading model resources...")
def load_resources():
    required_files = {
        "tfidf": MODEL_DIR / "tfidf_global.pkl",
        "passthrough": MODEL_DIR / "passthrough_map.json",
        "geometry": MODEL_DIR / "species_geometry.pkl",
    }

    missing = [str(path) for path in required_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))

    tfidf = joblib.load(required_files["tfidf"])

    with required_files["passthrough"].open(encoding="utf-8") as file:
        passthrough_map = json.load(file)

    geometry_payload = joblib.load(required_files["geometry"])

    medoid_references_path = MODEL_DIR / "species_medoid_references.pkl"
    if medoid_references_path.exists():
        medoid_payload = joblib.load(medoid_references_path)
    else:
        medoid_payload = {"references": {}, "n_references": 0}

    try:
        import variables

        thresholds = variables.THRESHOLDS
        kmer_size = variables.K_MER_SIZE
    except (ImportError, AttributeError):
        thresholds = DEFAULT_THRESHOLDS
        kmer_size = 6

    return {
        "tfidf": tfidf,
        "passthrough": passthrough_map,
        "thresholds": thresholds,
        "kmer_size": kmer_size,
        "geometry": geometry_payload["species"],
        "geometry_metadata": geometry_payload,
        "medoid_references": medoid_payload["references"],
        "medoid_metadata": medoid_payload,
        "separation_threshold": float(
            geometry_payload.get("separation_threshold", 1.0)
        ),
    }


#BLASTN-like scoring parameters for local pairwise alignment
ALIGNER = Align.PairwiseAligner()
ALIGNER.mode = "local"
ALIGNER.match_score = 2.0
ALIGNER.mismatch_score = -3.0
ALIGNER.open_gap_score = -5.0
ALIGNER.extend_gap_score = -2.0
ALIGNER.wildcard = "N"


try:
    RESOURCES = load_resources()
except Exception as error:
    st.error(f"Unable to load the classifier: {error}")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_local_model(level, parent_value=None):
    """Load one LCPN node and its label encoder."""
    if level == "phylum":
        model_file = MODEL_DIR / "xgb_root.pkl"
        encoder_file = MODEL_DIR / "le_root.pkl"
    else:
        if parent_value is None:
            return None, None

        safe_parent = re.sub(r"[^a-zA-Z0-9_]", "_", str(parent_value))
        model_file = MODEL_DIR / f"xgb_{level}_{safe_parent}.pkl"
        encoder_file = MODEL_DIR / f"le_{level}_{safe_parent}.pkl"

    if not model_file.exists() or not encoder_file.exists():
        return None, None

    return joblib.load(model_file), joblib.load(encoder_file)


#=============================================================================
# SEQUENCE PROCESSING AND HIERARCHICAL PREDICTION
#=============================================================================
def sequence_to_kmers(sequence, kmer_size):
    """Create alignment-free k-mers, excluding ambiguous windows."""
    cleaned = "".join(
        base if base in {"A", "C", "G", "T"} else "0"
        for base in sequence.upper()
    )

    return " ".join(
        cleaned[index : index + kmer_size]
        for index in range(len(cleaned) - kmer_size + 1)
        if "0" not in cleaned[index : index + kmer_size]
    )


def get_top_candidates(probabilities, label_encoder, top_k=5):
    """Return the local Top-k classes of a species model."""
    ranked_indices = np.argsort(probabilities)[::-1][:top_k]
    candidates = []

    for rank, class_index in enumerate(ranked_indices, start=1):
        species = label_encoder.inverse_transform([int(class_index)])[0]
        candidates.append(
            {
                "rank": rank,
                "species": str(species),
                "probability": float(probabilities[class_index]),
            }
        )

    return candidates


def predict_sequence(sequence):
    """Run the LCPN classifier from phylum to species."""
    kmer_text = sequence_to_kmers(sequence, RESOURCES["kmer_size"])
    if not kmer_text:
        return []

    vector = RESOURCES["tfidf"].transform([kmer_text])
    path = []
    current_parent = None

    for level_index, level in enumerate(LEVELS):
        # A passthrough is used only when the parent has one taxonomic child.
        if level_index > 0:
            edge = f"{LEVELS[level_index - 1]}->{level}"
            passthrough_value = RESOURCES["passthrough"].get(edge, {}).get(
                current_parent
            )

            if passthrough_value is not None:
                path.append(
                    {
                        "level": level,
                        "label": str(passthrough_value),
                        "confidence": None,
                        "status": "PASSTHROUGH",
                        "top_k": [],
                    }
                )
                current_parent = passthrough_value
                continue

        model, label_encoder = load_local_model(level, current_parent)

        if model is None:
            path.append(
                {
                    "level": level,
                    "label": None,
                    "confidence": None,
                    "status": "NO_MODEL",
                    "top_k": [],
                }
            )
            break

        probabilities = model.predict_proba(vector)[0]
        best_index = int(np.argmax(probabilities))
        best_label = label_encoder.inverse_transform([best_index])[0]
        confidence = float(probabilities[best_index])

        top_k = (
            get_top_candidates(probabilities, label_encoder)
            if level == "species"
            else []
        )

        status = (
            "ACCEPTED"
            if confidence >= RESOURCES["thresholds"][level]
            else "BELOW_THRESHOLD"
        )

        path.append(
            {
                "level": level,
                "label": str(best_label),
                "confidence": confidence,
                "status": status,
                "top_k": top_k,
            }
        )

        if status == "BELOW_THRESHOLD":
            break

        current_parent = best_label

    return path


def get_final_prediction(path):
    """Return the deepest accepted or passthrough prediction."""
    for step in reversed(path):
        if step["status"] in {"ACCEPTED", "PASSTHROUGH"}:
            return step["level"], step["label"]
    return None, None


def get_species_decision(path):
    """Return the species-model decision, including a rejected Top-1."""
    for step in path:
        if step["level"] == "species" and step["top_k"]:
            return step
    return None


def format_taxon(label):
    return str(label).replace("_", " ") if label else ""


def get_stop_reason(path):
    if not path:
        return "no_valid_kmers"

    last_step = path[-1]
    if last_step["status"] == "BELOW_THRESHOLD":
        return f"below_{last_step['level']}_threshold"
    if last_step["status"] == "NO_MODEL":
        return f"no_{last_step['level']}_model"
    return "completed"


#=============================================================================
# ALIGNMENT-FREE SPECIES SEPARATION
#=============================================================================
def interpret_separation(ratio):
    if ratio < 0.5:
        return "strong overlap"
    if ratio < RESOURCES["separation_threshold"]:
        return "partial overlap"
    return "separated"


def calculate_pair_separation(species_a, species_b):
    """
    Calculate centroid distance divided by the sum of median radii.

    All quantities use cosine distance in the training TF-IDF space.
    """
    geometry_a = RESOURCES["geometry"].get(species_a)
    geometry_b = RESOURCES["geometry"].get(species_b)

    if geometry_a is None or geometry_b is None:
        return None

    centroid_a = np.asarray(geometry_a["centroid"]).reshape(1, -1)
    centroid_b = np.asarray(geometry_b["centroid"]).reshape(1, -1)
    centroid_distance = float(cosine_distances(centroid_a, centroid_b)[0, 0])

    radius_a = float(geometry_a["median_radius"])
    radius_b = float(geometry_b["median_radius"])
    denominator = radius_a + radius_b

    if denominator <= np.finfo(float).eps:
        ratio = float("inf") if centroid_distance > 0 else 0.0
    else:
        ratio = centroid_distance / denominator

    return {
        "centroid_distance": centroid_distance,
        "separation_ratio": float(ratio),
        "separation_threshold": RESOURCES["separation_threshold"],
        "limited_separation": bool(
            ratio < RESOURCES["separation_threshold"]
        ),
        "interpretation": interpret_separation(ratio),
        "species_a_n_train": int(geometry_a["n_train"]),
        "species_b_n_train": int(geometry_b["n_train"]),
        "species_a_median_radius": radius_a,
        "species_b_median_radius": radius_b,
    }


def build_separation_rows(sequence_id, path):
    """Build one CSV row for every Top-1 versus alternative pair."""
    species_step = get_species_decision(path)
    if species_step is None or len(species_step["top_k"]) < 2:
        return []

    reached_level, final_prediction = get_final_prediction(path)
    top_1 = species_step["top_k"][0]
    top_1_accepted = species_step["status"] == "ACCEPTED"
    rows = []

    for alternative in species_step["top_k"][1:]:
        separation = calculate_pair_separation(
            top_1["species"], alternative["species"]
        )
        if separation is None:
            continue

        rows.append(
            {
                "sequence_id": sequence_id,
                "reached_level": reached_level or "none",
                "final_prediction": (
                    format_taxon(final_prediction)
                    if final_prediction
                    else "unclassified"
                ),
                "top1_species_accepted": top_1_accepted,
                "top1_species": format_taxon(top_1["species"]),
                "top1_probability": top_1["probability"],
                "top1_n_train": separation["species_a_n_train"],
                "top1_median_radius": separation[
                    "species_a_median_radius"
                ],
                "alternative_rank": alternative["rank"],
                "alternative_species": format_taxon(
                    alternative["species"]
                ),
                "alternative_probability": alternative["probability"],
                "alternative_n_train": separation["species_b_n_train"],
                "alternative_median_radius": separation[
                    "species_b_median_radius"
                ],
                "centroid_distance": separation["centroid_distance"],
                "separation_ratio": separation["separation_ratio"],
                "separation_threshold": separation[
                    "separation_threshold"
                ],
                "limited_separation": separation["limited_separation"],
                "interpretation": separation["interpretation"],
            }
        )

    return rows


def build_candidate_rows(sequence_id, path):
    """Build one readable row for every local species candidate."""
    species_step = get_species_decision(path)
    if species_step is None:
        return []

    reached_level, final_prediction = get_final_prediction(path)
    top_1_accepted = species_step["status"] == "ACCEPTED"
    rows = []

    for candidate in species_step["top_k"]:
        geometry = RESOURCES["geometry"].get(candidate["species"], {})
        rows.append(
            {
                "sequence_id": sequence_id,
                "reached_level": reached_level or "none",
                "final_prediction": (
                    format_taxon(final_prediction)
                    if final_prediction
                    else "unclassified"
                ),
                "top1_species_accepted": top_1_accepted,
                "rank": candidate["rank"],
                "species": format_taxon(candidate["species"]),
                "probability": candidate["probability"],
                "n_train": geometry.get("n_train"),
            }
        )

    return rows


def top_k_as_text(path):
    species_step = get_species_decision(path)
    if species_step is None:
        return ""

    return "; ".join(
        f"{format_taxon(candidate['species'])} "
        f"({candidate['probability'] * 100:.1f}%)"
        for candidate in species_step["top_k"]
    )


def separation_as_text(rows):
    return "; ".join(
        f"{row['top1_species']} vs {row['alternative_species']}: "
        f"{row['separation_ratio']:.2f} ({row['interpretation']})"
        for row in rows
    )


#=============================================================================
# LOCAL ALIGNMENT AGAINST SPECIES MEDOIDS
#=============================================================================
def normalize_alignment_sequence(sequence):
    """Keep ACGT and map every ambiguous symbol to N"""
    return "".join(
        base if base in {"A", "C", "G", "T"} else "N"
        for base in str(sequence).upper()
    )


def reverse_complement(sequence):
    translation = str.maketrans("ACGTN", "TGCAN")
    return sequence.translate(translation)[::-1]


def local_alignment_metrics(query_sequence, reference_sequence):
    """
    Align a query locally against one medoid, testing both query strands.
    Percent identity includes gap columns in the alignment denominator.
    Query coverage is the number of query bases participating in the local
    alignment divided by the complete query length
    """
    query = normalize_alignment_sequence(query_sequence)
    reference = normalize_alignment_sequence(reference_sequence)

    if not query or not reference:
        return None

    orientation_results = []

    for strand, oriented_query in (
        ("+", query),
        ("-", reverse_complement(query)),
    ):
        alignments = ALIGNER.align(reference, oriented_query)

        try:
            alignment = alignments[0]
        except (IndexError, OverflowError):
            continue

        aligned_blocks = alignment.aligned
        reference_blocks = aligned_blocks[0]
        query_blocks = aligned_blocks[1]

        identities = 0
        valid_blocks = True

        for (ref_start, ref_end), (query_start, query_end) in zip(
            reference_blocks,
            query_blocks,
        ):
            block_length = int(ref_end - ref_start)

            reference_block = reference[int(ref_start) : int(ref_end)]
            query_block = oriented_query[int(query_start) : int(query_end)]

            if block_length != len(query_block):
                valid_blocks = False
                break

            identities += sum(
                reference_base == query_base
                and reference_base in {"A", "C", "G", "T"}
                for reference_base, query_base in zip(
                    reference_block,
                    query_block,
                )
            )

        if not valid_blocks:
            continue

        alignment_length = int(alignment.length)
        if alignment_length == 0:
            continue

        query_coordinates = np.asarray(alignment.coordinates[1])
        query_aligned_bases = int(
            query_coordinates.max() - query_coordinates.min()
        )
        identity = identities / alignment_length
        query_coverage = query_aligned_bases / len(query)

        orientation_results.append(
            {
                "strand": strand,
                "score": float(alignment.score),
                "identity": float(identity),
                "query_coverage": float(query_coverage),
                "identities": int(identities),
                "alignment_length": alignment_length,
                "query_aligned_bases": int(query_aligned_bases),
            }
        )

    if not orientation_results:
        return None

    return max(
        orientation_results,
        key=lambda result: (
            result["score"],
            result["query_coverage"],
            result["identity"],
        ),
    )


def build_medoid_alignment_rows(
    query_sequence,
    path,
    sequence_id="single_sequence",
):
    """Align the query against the medoids of its local Top-k candidates"""
    species_step = get_species_decision(path)
    if species_step is None:
        return []

    reached_level, final_prediction = get_final_prediction(path)
    top_1_accepted = species_step["status"] == "ACCEPTED"
    rows = []

    for candidate in species_step["top_k"]:
        reference = RESOURCES["medoid_references"].get(candidate["species"])
        if reference is None:
            continue

        metrics = local_alignment_metrics(
            query_sequence,
            reference["sequence"],
        )
        if metrics is None:
            continue

        rows.append(
            {
                "sequence_id": sequence_id,
                "reached_level": reached_level or "none",
                "final_prediction": (
                    format_taxon(final_prediction)
                    if final_prediction
                    else "unclassified"
                ),
                "top1_species_accepted": top_1_accepted,
                "rank": candidate["rank"],
                "species": format_taxon(candidate["species"]),
                "ai_probability": candidate["probability"],
                "query_coverage": metrics["query_coverage"],
                "percent_identity": metrics["identity"],
                "alignment_length": metrics["alignment_length"],
                "alignment_score": metrics["score"],
                "strand": metrics["strand"],
                "medoid_id": reference["medoid_id"],
                "medoid_length_bp": reference["length_bp"],
            }
        )

    return rows


#=============================================================================
# FASTA AND BATCH OUTPUT
#=============================================================================
def parse_fasta(text):
    """Parse FASTA text while preserving repeated sequence identifiers"""
    records = []
    current_id = None
    sequence_parts = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith(">"):
            if current_id is not None:
                records.append((current_id, "".join(sequence_parts)))

            header = line[1:].strip()
            current_id = (
                header.split()[0]
                if header
                else f"sequence_{len(records) + 1}"
            )
            sequence_parts = []
        elif current_id is not None:
            sequence_parts.append(line.upper())

    if current_id is not None:
        records.append((current_id, "".join(sequence_parts)))

    return records


def build_classification_row(sequence_id, sequence, path):
    reached_level, final_prediction = get_final_prediction(path)
    species_step = get_species_decision(path)

    row = {
        "sequence_id": sequence_id,
        "length_bp": len(sequence),
        "outside_training_length_range": not (
            TRAINING_MIN_LENGTH <= len(sequence) <= TRAINING_MAX_LENGTH
        ),
        "deepest_level": reached_level or "none",
        "best_prediction": (
            format_taxon(final_prediction)
            if final_prediction
            else "unclassified"
        ),
        "stop_reason": get_stop_reason(path),
    }

    for level in LEVELS:
        step = next(
            (item for item in path if item["level"] == level), None
        )
        accepted = step and step["status"] in {"ACCEPTED", "PASSTHROUGH"}

        row[level] = format_taxon(step["label"]) if accepted else ""
        row[f"confidence_{level}"] = (
            round(step["confidence"], 6)
            if step and step["confidence"] is not None
            else ""
        )
        row[f"status_{level}"] = step["status"] if step else "NOT_REACHED"

    row["species_candidates_available"] = species_step is not None
    row["top1_species_accepted"] = bool(
        species_step and species_step["status"] == "ACCEPTED"
    )
    row["top5_species"] = top_k_as_text(path)

    return row


def classify_fasta_records(records, include_medoid_alignments=False):
    result_rows = []
    candidate_rows = []
    separation_rows = []
    alignment_rows = []
    progress = st.progress(0.0)
    message = st.empty()

    for index, (sequence_id, raw_sequence) in enumerate(records, start=1):
        message.caption(
            f"Classifying {index}/{len(records)}: {sequence_id[:50]}"
        )

        sequence = re.sub(r"\s+", "", raw_sequence.upper())
        path = predict_sequence(sequence) if len(sequence) >= 50 else []

        result_row = build_classification_row(sequence_id, sequence, path)
        if len(sequence) < 50:
            result_row["stop_reason"] = "sequence_too_short"

        current_separations = build_separation_rows(sequence_id, path)
        current_candidates = build_candidate_rows(sequence_id, path)
        current_alignments = (
            build_medoid_alignment_rows(
                sequence,
                path,
                sequence_id=sequence_id,
            )
            if include_medoid_alignments
            else []
        )
        result_row["top1_vs_alternatives"] = separation_as_text(
            current_separations
        )

        result_rows.append(result_row)
        candidate_rows.extend(current_candidates)
        separation_rows.extend(current_separations)
        alignment_rows.extend(current_alignments)
        progress.progress(index / len(records))

    progress.empty()
    message.empty()

    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(candidate_rows),
        pd.DataFrame(separation_rows),
        pd.DataFrame(alignment_rows),
    )


#=============================================================================
# UI HELPERS
#=============================================================================
def path_dataframe(path):
    rows = []

    for step in path:
        threshold = (
            RESOURCES["thresholds"][step["level"]]
            if step["confidence"] is not None
            else None
        )
        rows.append(
            {
                "Level": step["level"],
                "Prediction": format_taxon(step["label"]),
                "Confidence (%)": (
                    step["confidence"] * 100
                    if step["confidence"] is not None
                    else None
                ),
                "Threshold (%)": threshold * 100 if threshold else None,
                "Status": step["status"],
            }
        )

    return pd.DataFrame(rows)


def show_species_analysis(path, query_sequence):
    species_step = get_species_decision(path)

    if species_step is None:
        passthrough_species = any(
            step["level"] == "species" and step["status"] == "PASSTHROUGH"
            for step in path
        )
        if passthrough_species:
            st.info(
                "Species was reached through a single-child passthrough. "
                "No alternative-species ranking is available."
            )
        else:
            st.info(
                "No multi-species classifier was reached. Top-5 species and "
                "pairwise separation are therefore unavailable."
            )
        return

    rows = build_separation_rows("single_sequence", path)

    if species_step["status"] == "ACCEPTED":
        st.success("The Top-1 species passed the calibrated species threshold.")

        limited_pairs = [
            row
            for row in rows
            if row["separation_ratio"] < row["separation_threshold"]
        ]
        if limited_pairs:
            pair_list = ", ".join(
                f"{row['alternative_species']} "
                f"(ratio {row['separation_ratio']:.2f})"
                for row in limited_pairs
            )
            st.warning(
                "High-confidence prediction with limited species separability. "
                "The Top-1 species was accepted, but its median region is not "
                f"separated from: {pair_list}. This does not mean that the "
                "prediction is incorrect; it indicates structural ambiguity "
                "among these species in the alignment-free representation."
            )
    else:
        st.warning(
            "The Top-1 species did not pass the species threshold. These "
            "candidates are exploratory; the final accepted result remains "
            "at the preceding taxonomic level."
        )

    candidate_rows = []
    for candidate in species_step["top_k"]:
        geometry = RESOURCES["geometry"].get(candidate["species"], {})
        candidate_rows.append(
            {
                "Rank": candidate["rank"],
                "Species": format_taxon(candidate["species"]),
                "Probability (%)": candidate["probability"] * 100,
                "Training sequences": geometry.get("n_train"),
            }
        )

    st.subheader("Local Top-5 species candidates")
    st.caption(
        "The candidates belong only to the species classifier of the reached "
        "genus; this is not a ranking across every fungal species."
    )
    st.dataframe(pd.DataFrame(candidate_rows), hide_index=True)

    if not rows:
        st.info("Separation values are unavailable for these candidates.")
        return

    separation_table = pd.DataFrame(rows)[
        [
            "alternative_rank",
            "alternative_species",
            "alternative_probability",
            "separation_ratio",
            "interpretation",
        ]
    ].rename(
        columns={
            "alternative_rank": "Rank",
            "alternative_species": "Alternative species",
            "alternative_probability": "Probability (%)",
            "separation_ratio": "Separation ratio",
            "interpretation": "Interpretation",
        }
    )
    separation_table["Probability (%)"] *= 100

    st.subheader("Top-1 pairwise separation")
    st.caption(
        "Ratio = centroid distance / sum of the two median within-species radii."
    )
    st.dataframe(separation_table, hide_index=True)

    with st.expander("How to interpret the ratio"):
        st.markdown(
            """
            - **Ratio < 0.50:** strong overlap in the model representation.
            - **0.50 ≤ ratio < 1.00:** partial overlap in the model representation.
            - **Ratio ≥ 1.00:** median regions are separated.

            This is an alignment-free diagnostic in TF-IDF model space. It is
            not, by itself, proof of biological identity or a formal barcoding gap.
            """
        )

    st.subheader("Local alignment against species medoids")
    st.caption(
        "BLAST-like view: the query is locally aligned against one real "
        "training medoid for each Top-k species candidate. Candidate order "
        "remains the order produced by the AI model."
    )

    alignment_rows = build_medoid_alignment_rows(query_sequence, path)
    if not alignment_rows:
        st.info(
            "Medoid sequence references are not available. Add "
            "models_hier/species_medoid_references.pkl to enable this panel."
        )
        return

    alignment_table = pd.DataFrame(alignment_rows).rename(
        columns={
            "rank": "Rank",
            "species": "Species",
            "ai_probability": "AI probability (%)",
            "query_coverage": "Query cover (%)",
            "percent_identity": "Percent identity (%)",
            "alignment_length": "Alignment length",
            "alignment_score": "Local score",
            "strand": "Strand",
            "medoid_id": "Medoid reference ID",
            "medoid_length_bp": "Reference length",
        }
    )
    alignment_table["AI probability (%)"] *= 100
    alignment_table["Query cover (%)"] *= 100
    alignment_table["Percent identity (%)"] *= 100

    st.dataframe(
        alignment_table[
            [
                "Rank",
                "Species",
                "AI probability (%)",
                "Query cover (%)",
                "Percent identity (%)",
                "Alignment length",
                "Local score",
                "Strand",
                "Medoid reference ID",
                "Reference length",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "AI probability (%)": st.column_config.NumberColumn(format="%.2f"),
            "Query cover (%)": st.column_config.NumberColumn(format="%.2f"),
            "Percent identity (%)": st.column_config.NumberColumn(format="%.2f"),
            "Local score": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.info(
        "This panel is not a BLAST database search: it compares the query "
        "only with the medoid of each AI candidate and therefore does not "
        "report E-values or bit scores. Identity should always be interpreted "
        "together with query coverage."
    )

    with st.expander("Alignment method"):
        st.markdown(
            """
            - Local pairwise alignment on both DNA orientations.
            - Match reward: +2; mismatch penalty: −3.
            - Gap-open penalty: −5; gap-extension penalty: −2.
            - Percent identity includes gap columns in the alignment length.
            - Query coverage is the aligned query span divided by query length.
            - The reference is a real training sequence selected as the medoid
            in TF-IDF 6-mer cosine space; it is not a biological type sequence.
            """
        )


def show_batch_results(results, candidates, separations, alignments):
    st.subheader("Results summary")

    column_1, column_2, column_3, column_4 = st.columns(4)
    column_1.metric("Sequences", len(results))
    column_2.metric(
        "Accepted to species",
        int((results["deepest_level"] == "species").sum()),
    )
    column_3.metric(
        "Accepted to genus",
        int((results["deepest_level"] == "genus").sum()),
    )
    column_4.metric(
        "Unclassified",
        int((results["deepest_level"] == "none").sum()),
    )

    depth_map = {level: index + 1 for index, level in enumerate(LEVELS)}
    average_depth = results["deepest_level"].map(depth_map).fillna(0).mean()
    st.write(f"Average accepted depth: **{average_depth:.2f}/6**")

    outside_count = int(results["outside_training_length_range"].sum())
    if outside_count:
        st.warning(
            f"{outside_count} sequence(s) are outside the 200–1,000 bp "
            "length range used for model training. Their predictions should "
            "be interpreted with caution."
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    top_k_tab, predictions_tab, downloads_tab = st.tabs(
        ["Top-k species", "All predictions", "Downloads"]
    )

    with top_k_tab:
        st.subheader("Explore local species candidates")
        st.caption(
            "Select a sequence to inspect all candidates returned by its local "
            "species classifier."
        )

        sequence_ids = results["sequence_id"].tolist()
        selected_id = st.selectbox("Sequence", sequence_ids)
        selected_result = results[results["sequence_id"] == selected_id].iloc[0]

        result_column, level_column, status_column = st.columns(3)
        result_column.metric(
            "Final accepted prediction",
            selected_result["best_prediction"],
        )
        level_column.metric(
            "Deepest accepted level",
            selected_result["deepest_level"],
        )

        selected_candidates = (
            candidates[candidates["sequence_id"] == selected_id].copy()
            if not candidates.empty
            else pd.DataFrame()
        )

        if selected_candidates.empty:
            status_column.metric("Species candidate status", "Not available")
            st.info(
                "This sequence did not reach a local multi-species classifier. "
                "The hierarchical result above is still the final accepted result."
            )
        else:
            top_1_accepted = bool(
                selected_candidates.iloc[0]["top1_species_accepted"]
            )
            status_column.metric(
                "Species candidate status",
                "Accepted" if top_1_accepted else "Exploratory",
            )

            if top_1_accepted:
                st.success(
                    "The first species candidate passed the calibrated threshold."
                )
            else:
                st.warning(
                    "The Top-1 species did not pass the species threshold. The "
                    "ranking is exploratory; the final accepted prediction shown "
                    "above remains unchanged."
                )

            candidate_table = selected_candidates[
                ["rank", "species", "probability", "n_train"]
            ].rename(
                columns={
                    "rank": "Rank",
                    "species": "Species",
                    "probability": "Probability (%)",
                    "n_train": "Training sequences",
                }
            )
            candidate_table["Probability (%)"] *= 100

            st.markdown("**Top-k candidates**")
            st.dataframe(
                candidate_table,
                width="stretch",
                hide_index=True,
                column_config={
                    "Probability (%)": st.column_config.NumberColumn(
                        format="%.2f"
                    )
                },
            )

            selected_separations = (
                separations[separations["sequence_id"] == selected_id].copy()
                if not separations.empty
                else pd.DataFrame()
            )

            if selected_separations.empty:
                st.info("Pairwise separation is unavailable for these candidates.")
            else:
                separation_table = selected_separations[
                    [
                        "alternative_rank",
                        "alternative_species",
                        "separation_ratio",
                        "interpretation",
                    ]
                ].rename(
                    columns={
                        "alternative_rank": "Rank",
                        "alternative_species": "Alternative species",
                        "separation_ratio": "Separation ratio",
                        "interpretation": "Interpretation",
                    }
                )

                st.markdown("**Top-1 versus alternatives**")
                st.dataframe(
                    separation_table,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Separation ratio": st.column_config.NumberColumn(
                            format="%.2f"
                        )
                    },
                )

            st.markdown("**Local alignment against species medoids**")
            selected_alignments = (
                alignments[alignments["sequence_id"] == selected_id].copy()
                if not alignments.empty
                else pd.DataFrame()
            )

            if selected_alignments.empty:
                st.info(
                    "Medoid alignments were not calculated for this sequence "
                    "or no local species candidates were available."
                )
            else:
                alignment_table = selected_alignments.rename(
                    columns={
                        "rank": "Rank",
                        "species": "Species",
                        "ai_probability": "AI probability (%)",
                        "query_coverage": "Query cover (%)",
                        "percent_identity": "Percent identity (%)",
                        "alignment_length": "Alignment length",
                        "alignment_score": "Local score",
                        "strand": "Strand",
                        "medoid_id": "Medoid reference ID",
                        "medoid_length_bp": "Reference length",
                    }
                )
                alignment_table["AI probability (%)"] *= 100
                alignment_table["Query cover (%)"] *= 100
                alignment_table["Percent identity (%)"] *= 100

                st.dataframe(
                    alignment_table[
                        [
                            "Rank",
                            "Species",
                            "AI probability (%)",
                            "Query cover (%)",
                            "Percent identity (%)",
                            "Alignment length",
                            "Local score",
                            "Strand",
                            "Medoid reference ID",
                            "Reference length",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "AI probability (%)": st.column_config.NumberColumn(
                            format="%.2f"
                        ),
                        "Query cover (%)": st.column_config.NumberColumn(
                            format="%.2f"
                        ),
                        "Percent identity (%)": st.column_config.NumberColumn(
                            format="%.2f"
                        ),
                        "Local score": st.column_config.NumberColumn(
                            format="%.1f"
                        ),
                    },
                )
                st.caption(
                    "BLAST-like local comparison against one training medoid "
                    "per AI candidate. This is not a BLAST database search and "
                    "does not provide E-values or bit scores."
                )

        if not candidates.empty:
            with st.expander("Show Top-k candidates for every sequence"):
                all_candidates = candidates.copy()
                all_candidates["probability"] *= 100
                all_candidates = all_candidates.rename(
                    columns={
                        "sequence_id": "Sequence",
                        "top1_species_accepted": "Top-1 accepted",
                        "rank": "Rank",
                        "species": "Species",
                        "probability": "Probability (%)",
                        "n_train": "Training sequences",
                    }
                )
                st.dataframe(
                    all_candidates[
                        [
                            "Sequence",
                            "Top-1 accepted",
                            "Rank",
                            "Species",
                            "Probability (%)",
                            "Training sequences",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

    with predictions_tab:
        st.subheader("Hierarchical predictions")
        preview_columns = [
            "sequence_id",
            "length_bp",
            "deepest_level",
            "best_prediction",
            "stop_reason",
            "top1_species_accepted",
        ]
        st.dataframe(
            results[preview_columns],
            width="stretch",
            hide_index=True,
        )

    with downloads_tab:
        st.write(
            "CSV downloads are optional and contain the complete numerical "
            "results for further analysis."
        )
        st.download_button(
            "Download classification results (CSV)",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name=f"FungiClass_results_{timestamp}.csv",
            mime="text/csv",
        )

        if not candidates.empty:
            st.download_button(
                "Download Top-k species candidates (CSV)",
                data=candidates.to_csv(index=False).encode("utf-8"),
                file_name=f"FungiClass_top_k_species_{timestamp}.csv",
                mime="text/csv",
            )

        if not separations.empty:
            st.download_button(
                "Download detailed species separation (CSV)",
                data=separations.to_csv(index=False).encode("utf-8"),
                file_name=f"FungiClass_species_separation_{timestamp}.csv",
                mime="text/csv",
            )

        if not alignments.empty:
            alignment_download = alignments.copy()
            alignment_download["ai_probability"] *= 100
            alignment_download["query_coverage"] *= 100
            alignment_download["percent_identity"] *= 100
            alignment_download = alignment_download.rename(
                columns={
                    "ai_probability": "ai_probability_percent",
                    "query_coverage": "query_coverage_percent",
                }
            )
            st.download_button(
                "Download medoid alignments (CSV)",
                data=alignment_download.to_csv(index=False).encode("utf-8"),
                file_name=f"FungiClass_medoid_alignments_{timestamp}.csv",
                mime="text/csv",
            )


#=============================================================================
# APPLICATION PAGES
#=============================================================================
st.title("🍄 FungiClass")
st.caption("Alignment-free hierarchical identification of fungal ITS sequences")

with st.sidebar:
    page = st.radio("Page", ["Single sequence", "Batch FASTA", "About"])
    st.divider()
    st.markdown(
        f"""
        <div class="sidebar-model-info">
            <div class="sidebar-model-row">
                <span class="info-pill">Model</span>
                <span>LCPN · XGBoost</span>
            </div>
            <div class="sidebar-model-row">
                <span class="info-pill">Marker</span>
                <span>fungal ITS</span>
            </div>
            <div class="sidebar-model-row">
                <span class="info-pill">Training</span>
                <span>
                    {RESOURCES['geometry_metadata']['n_training_sequences']:,}
                    sequences
                </span>
            </div>
            <div class="sidebar-model-row">
                <span class="info-pill">Species</span>
                <span>{EVALUABLE_SPECIES_COUNT:,} evaluable</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.warning("Research prototype — not validated for clinical diagnosis.")


if page == "Single sequence":
    st.header("Single-sequence identification")
    sequence_input = st.text_area(
        "Paste an ITS sequence",
        height=220,
        placeholder="ATCGGGTTAGCTATCGATCGATCGATCG...",
    )

    if st.button("Identify", type="primary"):
        lines = sequence_input.strip().splitlines()
        sequence = "".join(
            line.strip() for line in lines if not line.startswith(">")
        ).upper()
        sequence = re.sub(r"\s+", "", sequence)

        if not sequence:
            st.warning("Paste a sequence first.")
        elif len(sequence) < 50:
            st.warning("The sequence must contain at least 50 bases.")
        else:
            if not TRAINING_MIN_LENGTH <= len(sequence) <= TRAINING_MAX_LENGTH:
                st.warning(
                    "Sequence length is outside the 200–1,000 bp range used "
                    "for model training. Interpret the result with caution."
                )

            with st.spinner("Running hierarchical identification..."):
                path = predict_sequence(sequence)

            if not path:
                st.error("The sequence produced no valid k-mers.")
            else:
                reached_level, final_prediction = get_final_prediction(path)

                if final_prediction:
                    st.success(
                        f"Final accepted prediction: "
                        f"{format_taxon(final_prediction)} ({reached_level})"
                    )
                else:
                    st.warning("No taxonomic level was accepted.")

                st.subheader("Hierarchical path")
                st.dataframe(path_dataframe(path), hide_index=True)
                st.divider()
                show_species_analysis(path, sequence)


elif page == "Batch FASTA":
    st.header("Batch FASTA identification")
    uploaded_file = st.file_uploader(
        "Upload a FASTA file", type=["fasta", "fa", "fna", "txt"]
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_key = hashlib.sha256(file_bytes).hexdigest()
        fasta_text = file_bytes.decode("utf-8", errors="ignore")
        records = parse_fasta(fasta_text)

        if not records:
            st.error("No FASTA records were found.")
        else:
            st.info(f"Found {len(records)} sequences in {uploaded_file.name}.")

            medoid_alignments_available = bool(
                RESOURCES["medoid_references"]
            )
            include_medoid_alignments = st.checkbox(
                "Include local alignment against species medoids",
                value=medoid_alignments_available,
                disabled=not medoid_alignments_available,
                help=(
                    "For each sequence that reaches a local species classifier, "
                    "align the query on both strands against one training medoid "
                    "for every Top-k candidate. This adds identity and query "
                    "coverage but makes large batches slower."
                ),
            )

            if not medoid_alignments_available:
                st.warning(
                    "Medoid references are unavailable. Add "
                    "models_hier/species_medoid_references.pkl to enable "
                    "batch alignments."
                )
            elif include_medoid_alignments and len(records) > 500:
                st.warning(
                    "Medoid alignment is enabled for a large FASTA file. "
                    "Processing may take substantially longer."
                )

            analysis_key = (
                f"{file_key}:medoid_alignments="
                f"{int(include_medoid_alignments)}"
            )

            if st.session_state.get("batch_file_key") != analysis_key:
                st.session_state.pop("batch_results", None)
                st.session_state.pop("batch_candidates", None)
                st.session_state.pop("batch_separations", None)
                st.session_state.pop("batch_alignments", None)

            if st.button("Run batch identification", type="primary"):
                (
                    results,
                    candidates,
                    separations,
                    alignments,
                ) = classify_fasta_records(
                    records,
                    include_medoid_alignments=include_medoid_alignments,
                )
                st.session_state["batch_file_key"] = analysis_key
                st.session_state["batch_results"] = results
                st.session_state["batch_candidates"] = candidates
                st.session_state["batch_separations"] = separations
                st.session_state["batch_alignments"] = alignments

            if (
                st.session_state.get("batch_file_key") == analysis_key
                and "batch_results" in st.session_state
                and "batch_candidates" in st.session_state
                and "batch_separations" in st.session_state
                and "batch_alignments" in st.session_state
            ):
                show_batch_results(
                    st.session_state["batch_results"],
                    st.session_state["batch_candidates"],
                    st.session_state["batch_separations"],
                    st.session_state["batch_alignments"],
                )


else:
    st.header("About FungiClass")
    st.markdown(
        """
        FungiClass is an alignment-free classifier for fungal ITS sequences.
        It follows a **Local Classifier Per Parent Node (LCPN)** architecture
        through six taxonomic levels. Multi-child nodes use XGBoost models;
        single-child nodes use deterministic passthroughs.

        A prediction is accepted only if its confidence reaches the threshold
        selected on an independent calibration set. Otherwise, the system stops
        and returns the deepest accepted taxonomic level.

        When a local species classifier is reached, the application also reports
        the Top-5 candidates and an alignment-free pairwise separation ratio:

        `centroid distance / (median radius A + median radius B)`

        The measure uses cosine distance in TF-IDF 6-mer space. It is inspired
        by the pairwise geometry of the Davies–Bouldin index, using an inverted
        direction and robust median radii.

        As a complementary sequence-level check, each Top-5 candidate is also
        compared with one real training medoid through local pairwise alignment
        on both DNA orientations. The app reports percent identity and query
        coverage in a BLAST-like table, without presenting the comparison as a
        BLAST database search.
        """
    )

    st.subheader("Final model")
    model_summary = pd.DataFrame(
        {
            "Component": [
                "Representation",
                "Architecture",
                "Training length range",
                "Training sequences",
                "Species classifiers",
                "Modelled species classes",
                "Species medoid references",
                "Processable test sequences",
            ],
            "Value": [
                "TF-IDF of alignment-free 6-mers",
                "LCPN with XGBoost",
                "200–1,000 bp",
                "33,405",
                "155",
                "654",
                "654",
                "4,813",
            ],
        }
    )
    st.dataframe(model_summary, hide_index=True)

    st.subheader("Overall test performance")
    st.caption(
        "Selective accuracy is computed only over predictions accepted at a "
        "level. Coverage is computed over all 4,813 processable test sequences."
    )
    overall_performance = pd.DataFrame(
        {
            "Level": ["Phylum", "Class", "Order", "Family", "Genus", "Species"],
            "Selective accuracy (%)": [97.75, 94.09, 92.64, 91.08, 89.08, 66.84],
            "Coverage (%)": [
                4404 / 4813 * 100,
                4147 / 4813 * 100,
                3929 / 4813 * 100,
                3554 / 4813 * 100,
                2884 / 4813 * 100,
                971 / 4813 * 100,
            ],
            "Accepted predictions": [4404, 4147, 3929, 3554, 2884, 971],
        }
    )
    st.dataframe(overall_performance, hide_index=True)

    st.subheader("Common and rare test strata")
    stratum_performance = pd.DataFrame(
        {
            "Stratum": ["Common", "Common", "Rare", "Rare"],
            "Level": ["Genus", "Species", "Genus", "Species"],
            "Selective accuracy (%)": [92.81, 92.25, 86.92, 13.42],
            "Coverage (%)": [74.65, 46.47, 53.78, 9.21],
        }
    )
    st.dataframe(stratum_performance, hide_index=True)
    st.caption(
        "Common: 1,416 test sequences from species with at least 10 samples. "
        "Rare: 3,397 processable test sequences, with one held-out sequence "
        "for species represented by 3–9 samples. Rare-species performance is "
        "strongly affected by the availability of trainable species models."
    )

    st.info(
        "FungiClass is a research prototype and is not validated for clinical diagnosis."
    )
