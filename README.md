# Topic Modeling Pipeline

This project creates an automated pipeline for uncovering hidden thematic structures in large text corpora, specifically demonstrating its capabilities on the 20 Newsgroups dataset. Leveraging the power of Natural Language Processing (NLP), the system utilizes Gensim's Latent Dirichlet Allocation (LDA) to analyze unstructured text. The process begins with a robust preprocessing stage that cleans the data—removing noise like emails, punctuation, and stopwords—and standardizes words through lemmatization to ensure high-quality input for the model.

Beyond simple topic extraction, the pipeline incorporates an intelligent evaluation mechanism to determine the optimal number of topics by maximizing Coherence Scores ($C_v$). It iteratively trains multiple models to find the "best fit" configuration. The results are then presented through rich visualizations, including generated Word Clouds for quick thematic interpretation and an interactive `pyLDAvis` dashboard that allows users to explore the relationships and term distributions across different topics dynamically.

## Setup

1.  Ensure you have Python 3.11 installed (or similar). The script was tested with Python 3.11 from Homebrew.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Pipeline

Run the main script:

```bash
python topic_modeling_pipeline.py
```

## Outputs

All results are saved in the `output/` directory:

*   `coherence_scores.csv`: Table of Coherence Scores for different numbers of topics (k).
*   `lda_visualization.html`: Interactive pyLDAvis visualization of the best topic model.
*   `plots/`: Directory containing Word Cloud images for each topic in the best model.
