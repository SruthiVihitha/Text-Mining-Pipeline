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
  <img width="345" height="270" alt="Screenshot 2025-12-26 at 11 30 19 AM" src="https://github.com/user-attachments/assets/673e03ab-b617-415c-abf7-39c0c2bed6fc" />

*   `lda_visualization.html`: Interactive pyLDAvis visualization of the best topic model.
  <img width="1512" height="982" alt="Screenshot 2025-12-26 at 11 32 42 AM" src="https://github.com/user-attachments/assets/8a1eab58-fcf9-4827-9f34-eae35bfd3bd4" />
  <img width="1512" height="982" alt="Screenshot 2025-12-26 at 11 29 13 AM" src="https://github.com/user-attachments/assets/98115b4a-8267-4cc7-83db-24a8df9ca7ed" />

*   `plots/`: Directory containing Word Cloud images for each topic in the best model.
<img width="794" height="429" alt="topic_0_wordcloud" src="https://github.com/user-attachments/assets/1f263c98-fc41-469b-b4ed-b43ba7e98ea6" />
<img width="794" height="429" alt="topic_1_wordcloud" src="https://github.com/user-attachments/assets/80a1deb8-4b33-4ddd-bbd8-92944ddc6bcf" />
<img width="794" height="429" alt="topic_2_wordcloud" src="https://github.com/user-attachments/assets/a5029d48-710e-41d9-8bd4-0f672ccf6690" />
<img width="794" height="429" alt="topic_3_wordcloud" src="https://github.com/user-attachments/assets/51635867-151d-4db0-93be-d637da161c2c" />
