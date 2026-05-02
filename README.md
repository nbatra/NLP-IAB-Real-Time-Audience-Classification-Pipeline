# NLP IAB Real-Time Audience Classification Pipeline

An end-to-end production ML system that classifies website domains into **IAB Content Taxonomy** categories using NLP and generates high-value audience segments from real-time programmatic advertising bidstream data.

> **Production lineage:** The core architecture and scoring methodology in this project were designed and deployed at a real AdTech company by the author and an amazing engineering team. The model was deployed to calculate user interests in real time on bidstreams running at **8 million QPS**, enabled the company to detect intent and win bids earlier than competitors, and drove a fundamental shift in the KV-store data storage approach — from materialized segment memberships to compact probability vectors, reducing storage footprint by 3-8x. This repository is a clean-room reimplementation with synthetic data for educational and portfolio purposes — the pipeline design, two-tier lookup strategy, decay math, and segment economics are all drawn from that production system.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Why This Matters Commercially](#why-this-matters-commercially)
- [System Architecture](#system-architecture)
- [Pipeline Overview](#pipeline-overview)
- [Notebooks](#notebooks)
- [Key Results & Visualizations](#key-results--visualizations)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Production Considerations](#production-considerations)
- [Author](#author)

---

## Problem Statement

Every time someone loads a webpage, a real-time auction (RTB) decides which ad to show — in under 100ms. Advertisers pay premiums to reach users whose browsing behavior signals genuine interest. The challenge: **how do you classify 200K+ domains into ~700 IAB categories, track 500M+ users in real time, and assign audience segments — all within a 10ms latency budget per bid request?**

This system solves that by decomposing the problem into four interconnected components:

| Component | What It Does | Timescale |
|---|---|---|
| **Domain Classification** | Multi-label NLP classification of website content into IAB categories | Batch (daily) |
| **Real-Time User Scoring** | Accumulate per-user IAB probability vectors from bidstream events | Streaming (sub-ms/event at 8M QPS) |
| **Temporal Decay** | Exponential time-decay with category-specific half-lives | Applied at read-time |
| **Segment Generation** | Threshold + top-K assignment to named audience segments | Periodic (hourly) |

---

## Why This Matters Commercially

- **2-5x CPM premiums** for well-classified audience segments vs. untargeted inventory
- A Travel advertiser pays significantly more to reach users reliably classified into "IAB-20: Travel" than to show ads to a random audience
- **Probability vector storage** (our approach) uses **3-8x less KV-store memory** than traditional materialized segment stores — ~600 bytes/user compressed vs. 5-20 KB/user
- **Real-time intent detection** spots user interest **hours to a full day before** competitors relying on batch-refreshed segment lists, winning impressions at lower clearing prices

---

## System Architecture

```
╔═══════════════════════════════════════════════════════════════════════════╗
║              IAB REAL-TIME AUDIENCE CLASSIFICATION PIPELINE              ║
╚═══════════════════════════════════════════════════════════════════════════╝

 COMPONENT 1: DOMAIN CLASSIFICATION (Batch — Daily/Weekly)
 ┌──────────────────────────────────────────────────────────────────────┐
 │  Domain Corpus ──► TF-IDF Vectorizer ──► Multi-Label Classifier     │
 │  (200K domains)    (20K features,        (CalibratedCV + OneVsRest  │
 │                     bigrams)              Logistic Regression)       │
 │                                              │                      │
 │                              ┌────────────────┘                     │
 │                              ▼                                      │
 │                   Domain → IAB Lookup Table                         │
 │                   expedia.com → {Travel: 0.91, Hotels: 0.68, ...}   │
 │                   ~120MB, fits in-memory on every bid node           │
 └──────────────────────────────────┬───────────────────────────────────┘
                                    │
 COMPONENT 2: REAL-TIME USER SCORING (Streaming — Per Bid Request)
 ┌──────────────────────────────────┼───────────────────────────────────┐
 │  Bidstream ──► Kafka ──► Bid Processor ──► Two-Tier Scoring         │
 │  (OpenRTB)     (by user_id)                                         │
 │                                                                     │
 │  Tier 1: Domain lookup table (dict hit, <1ms)                       │
 │  Tier 2: Real-time TF-IDF inference for unknown domains (~10ms)     │
 │           with in-memory cache for subsequent hits                   │
 └──────────────────────────────────┬───────────────────────────────────┘
                                    │
 COMPONENT 3: SCORE DECAY (Read-Time)
 ┌──────────────────────────────────┼───────────────────────────────────┐
 │  decayed_score = raw_score × e^(−λ × Δt)                           │
 │                                                                     │
 │  Category-specific half-lives:                                      │
 │    Shopping: 3 days  │  Travel: 5 days  │  Finance: 15 days         │
 │    Education: 30 days│  News: 2 days    │  Technology: 14 days      │
 └──────────────────────────────────┬───────────────────────────────────┘
                                    │
 COMPONENT 4: AUDIENCE SEGMENT GENERATION (Periodic — Hourly)
 ┌──────────────────────────────────┼───────────────────────────────────┐
 │  For each user:                                                     │
 │    1. Apply decay to all scores                                     │
 │    2. Threshold filter (min confidence)                              │
 │    3. Top-K category selection                                      │
 │    4. Map to named segments: "IAB Travel Enthusiast"                │
 │    5. Export to DSP targeting systems                                │
 └─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Overview

### Notebook 1 — Domain Classification

Builds the NLP classification engine that maps website domains to IAB categories:

1. **Synthetic corpus generation** — 500 domains across 23 IAB Tier-1 categories with realistic, category-specific vocabulary pools (production: replace with web crawler output)
2. **TF-IDF vectorization** — 20,000 features, unigrams + bigrams, sublinear TF scaling, with a worked example showing exact term-weight calculations
3. **Model comparison** — 4 classifiers evaluated via stratified 5-fold cross-validation:
   - Logistic Regression (OneVsRest + CalibratedClassifierCV)
   - SGD Classifier (linear SVM with hinge loss)
   - Random Forest
   - **XGBoost**
4. **Production model training** — Best performer trained on full data, calibrated for probability output
5. **Domain lookup table** — Pre-computed `{domain → {IAB_category: probability}}` mapping, the core artifact consumed by the real-time pipeline
6. **Artifact serialization** — Exports `domain_df.parquet`, `pipeline_artifacts.pkl`, and `classification_model.pkl` for cross-notebook dependency

### Notebook 2 — Scoring, Segments & Evaluation

Simulates the real-time pipeline and evaluates segment quality:

1. **Bidstream simulation** — 10,000 events across 500 users with power-law activity distributions, diurnal traffic patterns, and user-domain affinity modeling
2. **Two-tier real-time scoring** — Tier 1 (lookup table, <1ms) with Tier 2 fallback (live TF-IDF inference via `RealTimeClassifier` with in-memory caching for unknown domains)
3. **Exponential time-decay** — Category-specific half-lives reflecting real-world intent persistence
4. **Audience segment generation** — Top-K thresholded assignment to named segments with size and confidence analysis
5. **User journey trace** — End-to-end walkthrough of a single user's path from raw bid events through scoring, decay, and final segment assignment
6. **Quality metrics** — Segment coverage, user distribution, score density analysis, and threshold sensitivity curves
7. **Model interpretability** — TF-IDF feature importance showing which terms drive each IAB category prediction

---

## Key Results & Visualizations

The pipeline generates 9 production-quality diagnostic plots:

| Plot | What It Shows |
|---|---|
| ![Domain Corpus Overview](plots/01_domain_corpus_overview.png) | Category distribution across the domain corpus — monitors for class imbalance that degrades classifier performance |
| ![TF-IDF Worked Example](plots/02_tfidf_worked_example.png) | Exact term weights for a single domain — the debugging view when a domain is misclassified |
| ![Model Comparison](plots/03_model_comparison.png) | 4-model head-to-head on F1, AUC, training time, and inference latency — the production model selection gate |
| ![Confusion Matrix](plots/04_confusion_matrix.png) | Per-category precision/recall heatmap — identifies which IAB categories the classifier confuses |
| ![Bidstream Patterns](plots/05_bidstream_patterns.png) | Simulated traffic with diurnal and power-law patterns — what production monitoring dashboards look like |
| ![Decay Curves](plots/06_decay_curves.png) | How category-specific half-lives shape score retention — the key business insight for segment freshness |
| ![Audience Segments](plots/07_audience_segments.png) | Segment size distribution and score density — are segments large enough for programmatic scale? |
| ![Threshold Analysis](plots/08_threshold_analysis.png) | Precision vs. coverage trade-off at different confidence thresholds — the tuning knob for segment quality |
| ![Feature Importance](plots/09_feature_importance.png) | Top TF-IDF features per IAB category — interpretability for stakeholder buy-in and debugging |

---

## Technical Stack

| Layer | Technology | Why |
|---|---|---|
| **Text Vectorization** | TF-IDF (scikit-learn) | 50x cheaper than BERT with only 2-3% accuracy gap; no GPU required; interpretable feature weights |
| **Classification** | Calibrated Logistic Regression (OneVsRest) | Well-calibrated probability distributions across 700 categories; fast inference; production standard in AdTech |
| **Model Comparison** | XGBoost, Random Forest, SGD | Evaluated for completeness; LR wins on calibration quality and inference speed at this scale |
| **Bidstream Scoring** | Two-tier: dict lookup + live inference | O(1) for known domains (<1ms); graceful fallback for long-tail domains (~10ms with caching) |
| **Score Decay** | Exponential decay (e^−λt) | Smooth, continuous; category-specific half-lives; computed lazily at read-time (no batch job) |
| **Data Format** | Parquet (Apache Arrow) | Columnar, compressed, fast I/O — production standard for ML pipelines |
| **Serialization** | Pickle (cross-notebook artifacts) | Notebook-to-notebook dependency management; production would use MLflow/W&B |

**Production equivalents** (not in demo, but referenced in design):
- **Domain lookup store:** Redis / Aerospike (~120MB, fits in memory on every bid node)
- **User score store:** Redis Cluster / Aerospike (~300GB compressed for 500M users)
- **Stream processing:** Kafka (partitioned by user_id) → custom bid processor
- **Segment export:** DynamoDB (real-time) + S3 (batch partner feeds)

---

## Project Structure

```
NLP IAB Real-Time Audience Classification Pipeline/
│
├── notebooks/
│   ├── 01_domain_classification.ipynb          # NLP pipeline: data → TF-IDF → model → lookup table
│   └── 02_scoring_segments_evaluation.ipynb    # Bidstream sim → scoring → decay → segments → eval
│
├── artifacts/                                  # Serialized outputs from Notebook 1 → consumed by Notebook 2
│   ├── domain_df.parquet                       # Domain corpus with text and labels
│   ├── pipeline_artifacts.pkl                  # Config, lookup table, TF-IDF matrix, model metrics
│   └── classification_model.pkl                # TF-IDF vectorizer + production classifier
│
├── plots/                                      # 9 diagnostic PNGs generated by the notebooks
│   ├── 01_domain_corpus_overview.png
│   ├── 02_tfidf_worked_example.png
│   ├── 03_model_comparison.png
│   ├── 04_confusion_matrix.png
│   ├── 05_bidstream_patterns.png
│   ├── 06_decay_curves.png
│   ├── 07_audience_segments.png
│   ├── 08_threshold_analysis.png
│   └── 09_feature_importance.png
│
├── docs/                                       # System design reference documents
│   ├── system_overview.html                    # Full architecture deep-dive
│   ├── competitive_advantages.html             # Storage & real-time scoring economics
│   ├── step_1_data_engineering.html
│   ├── step_2_feature_engineering.html
│   ├── step_3_model_architecture.html
│   ├── step_4_model_training.html
│   └── step_5_production_deployment.html
│
├── iab_audience_generation.ipynb               # Original combined notebook (reference copy)
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd "NLP IAB Real-Time Audience Classification Pipeline"

# Create virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install numpy pandas scikit-learn scipy matplotlib seaborn xgboost pyarrow jupyter ipykernel

# Run notebooks in order
jupyter notebook notebooks/01_domain_classification.ipynb
# Then:
jupyter notebook notebooks/02_scoring_segments_evaluation.ipynb
```

**Execution order matters:** Notebook 2 depends on artifacts generated by Notebook 1. Run them sequentially.

### What Changes for Production

The notebooks are designed so that **only the data source changes** when moving to production:

| Demo (this repo) | Production |
|---|---|
| Synthetic 500-domain corpus with generated text | Web crawler output (Scrapy) for 200K+ real domains |
| `domain_df.parquet` loaded from disk | Crawl pipeline → S3 → training pipeline (Airflow/Dagster) |
| In-memory Python dict for domain lookup | Redis/Aerospike cluster replicated to all bid nodes |
| Simulated 10K bidstream events | Kafka stream of 50B+ daily OpenRTB bid requests |
| Pickle artifacts | MLflow model registry with versioning and A/B deployment |

Everything else — the TF-IDF pipeline, model training, calibration, two-tier scoring logic, decay math, and segment generation — runs identically.

---

## Production Considerations

### Scale Numbers (Real-World)

| Dimension | Volume | Latency Budget |
|---|---|---|
| Bid requests / day | 50 Billion | <100ms total (OpenRTB deadline) |
| Bid requests / second | 580K sustained, 2M+ peak | ~10ms for audience enrichment |
| Unique users | 500M active, 2B total pool | <10ms user score lookup |
| Domain lookup table | 200K entries, ~120MB | <2ms (in-memory hash map) |
| User profile storage | ~300GB compressed (500M users) | Sub-10ms read/write |

### Key Design Decisions

1. **TF-IDF over BERT** — 50x cheaper training/inference, only 2-3% accuracy gap, no GPU at bid time
2. **Pre-computed lookup over real-time inference** — O(1) hash map lookup vs. 5-10ms model inference per request
3. **Curated 200K domains over crawl-everything** — 85% bid volume coverage at 1/4 the cost with better quality control
4. **Per-event exponential decay over batch reduction** — Smooth, category-specific, no nightly batch job sweeping 500M users
5. **Probability vectors over materialized segments** — 3-8x less storage, 1 write per event (vs. 5-15), instant new segment activation

---

## Author

**Nipun Batra**

- Email: batranipun@gmail.com
- LinkedIn: [linkedin.com/in/nipunbatra](https://www.linkedin.com/in/nipunbatra/)

Built from real-world production experience designing and deploying IAB audience classification systems in programmatic advertising. The core pipeline — domain classification via TF-IDF, two-tier bid-time scoring, exponential decay with category-specific half-lives, and probability vector storage — was implemented at scale with a talented AdTech engineering team processing billions of daily bid requests.

---

## License

This project is released for educational and portfolio purposes. The synthetic data, code, and documentation are original work. No proprietary data, model weights, or trade secrets from any employer are included.
