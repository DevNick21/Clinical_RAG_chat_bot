# Clinical RAG Evaluation Scoring System (Updated Aug 2025)

## Overview

The Clinical RAG system evaluates AI-generated answers to clinical questions using a small set of focused, weighted components with a strong emphasis on factual medical accuracy. Scores are combined via a simple weighted sum (weights sum to 1.0) with category-specific pass thresholds.

## Scoring Components & Weights

### 1. Factual Accuracy (Weight: 0.8 / 80%)

**Purpose**: Validate medical correctness against patient facts (MIMIC-IV) and question intent.

- Category-aware validation with a unified method for most categories.
- Keeps detailed, bespoke scoring for Labs and Prescriptions (values/units, dosages/timing).
- Uses structured keyword/code checks where applicable; falls back to a basic medical score when needed.

**Calculation**: Rule- and pattern-based validation per category with strict emphasis on correctness.

### 2. Context Relevance (Weight: 0.1 / 10%)

**Purpose**: Ensure retrieved documents are relevant to the question.

- Measures alignment between the question and retrieved sources.
- Encourages correct routing and document selection.

**Calculation**: Relevance scoring across retrieved documents (supports embedding/keyword signals).

### 3. Semantic Similarity (Weight: 0.05 / 5%)

**Purpose**: Check clinical language/term alignment between expected medical concepts and the answer.

- Uses clinical embeddings for conceptual alignment (not a substitute for factual validation).
- Embedding caching is used to avoid redundant computations.

**Calculation**: Cosine similarity between embeddings of generated answer and expected clinical keywords.

### 4. Performance (Weight: 0.05 / 5%)

**Purpose**: Reward responsive, efficient retrieval and penalize slow or empty results.

- Starts at 1.0 and applies multiplicative penalties based on thresholds.
- Penalizes very slow searches and cases where no documents are found.

**Calculation**:

- Thresholds: moderate_search_time = 2.0s, slow_search_time = 5.0s
- Penalties: moderate_penalty = 0.2, slow_penalty = 0.4, no_docs_penalty = 0.3
- Score = 1.0 × (1 - penalty) for each condition, then clamped to [0, 1].

## Overall Score Formula

```python
overall_score = (
    0.80 * factual_accuracy
  + 0.10 * context_relevance
  + 0.05 * semantic_similarity
  + 0.05 * performance
)
```

No further normalization is applied (weights sum to 1.0).

## Pass Thresholds by Category

The evaluator uses category-specific pass thresholds optimized for practical medical QA:

- Header: 0.60
- Diagnoses: 0.65
- Procedures: 0.65
- Labs: 0.55
- Microbiology: 0.55
- Prescriptions: 0.60
- Comprehensive: 0.50
- Default: 0.60

## Design Notes

- Validation is consolidated into a single parameterized method for most categories; Labs and Prescriptions retain dedicated, detailed scoring.
- Embedding caching is enabled to speed up semantic similarity without changing behavior.
- Thresholds and weights reflect the current configuration and are tuned for factual accuracy dominance.

## Removed/Deprecated Components

- Behavior score (professional tone/disclaimer checks) — removed from scoring.
- Completeness score (entity coverage) — removed to reduce noise and complexity.
- Retrieval metrics (precision/recall/F1) — removed; context relevance suffices.
- Entity extraction outputs — removed from results and scoring.

## Alignment with Configuration

- Weights: `EVALUATION_SCORING_WEIGHTS = { 'factual_accuracy': 0.8, 'context_relevance': 0.1, 'semantic_similarity': 0.05, 'performance': 0.05 }`
- Performance thresholds: `EVALUATION_PERFORMANCE_THRESHOLDS = { 'moderate_search_time': 2.0, 'slow_search_time': 5.0, 'moderate_penalty': 0.2, 'slow_penalty': 0.4, 'no_docs_penalty': 0.3 }`
- Pass thresholds: as listed above (`EVALUATION_PASS_THRESHOLDS`).
