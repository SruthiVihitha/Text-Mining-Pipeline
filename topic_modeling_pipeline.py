import os
import re
import warnings
import pandas as pd
import numpy as np
import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    # Setup directories
    os.makedirs('output', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)

    # 1. Download NLTK resources
    print("Downloading NLTK resources...")
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/omw-1.4')
    except LookupError:
        nltk.download('omw-1.4')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

    # 2. Load Data and ensure it matches "CSV or JSON format" requirement
    print("Loading dataset...")
    # We will use 20newsgroups as our source of "Sample news articles"
    categories = ['sci.space', 'soc.religion.christian', 'comp.graphics', 'rec.sport.baseball']
    try:
        newsgroups = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # SAVE to JSON to strictly adhere to "Sample news articles in CSV or JSON format"
    # This simulates having the file provided in the documented format.
    df_temp = pd.DataFrame({'content': newsgroups.data, 'target': newsgroups.target})
    json_path = 'output/sample_news_articles.json'
    df_temp.to_json(json_path, orient='records')
    print(f"Created sample resource: {json_path}")

    # NOW LOAD from the JSON file to simulate the pipeline starting from the provided resource
    print(f"Reading from {json_path}...")
    df = pd.read_json(json_path)
    data = df['content'].tolist()
    
    # Determine the actual number of topics (ground truth)
    true_k = len(categories)
    print(f"Loaded {len(data)} documents. Ground truth topics: {true_k}")

    # 3. Preprocessing
    print("Preprocessing text...")
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])

    def preprocess(text):
        # Remove emails
        text = re.sub(r'\S*@\S*\s?', '', text)
        # Remove new line characters
        text = re.sub(r'\s+', ' ', text)
        # Remove single quotes
        text = re.sub(r"\'", "", text)
        
        # Tokenize using gensim simple_preprocess: converts to lowercase, removes punctuation
        # deacc=True removes punctuations
        tokens = gensim.utils.simple_preprocess(str(text), deacc=True)
        
        # Remove stopwords and short words
        tokens = [w for w in tokens if w not in stop_words and len(w) > 3]
        
        # Lemmatization
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens

    data_processed = [preprocess(doc) for doc in data]
    
    # 4. Dictionary and Corpus
    print("Building Dictionary and Corpus...")
    id2word = corpora.Dictionary(data_processed)
    
    # Filter extremes: occur in less than 10 documents or more than 50% of documents
    id2word.filter_extremes(no_below=10, no_above=0.5)
    
    corpus = [id2word.doc2bow(text) for text in data_processed]
    
    print(f"Dictionary size: {len(id2word)}")
    print(f"Corpus size: {len(corpus)}")

    # 5. Experiment with Topic Counts
    # We expect close to 4 topics. Let's try [3, 4, 5, 6]
    topic_counts = [3, 4, 5, 6]
    results = []
    models = {}

    print("Training LDA models and computing coherence scores...")
    for k in topic_counts:
        print(f"  Training for k={k}...")
        
        # Train LDA Model
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=k,
                                               random_state=100,
                                               chunksize=100,
                                               passes=10,
                                               per_word_topics=True)
        
        # Compute Coherence Score using c_v
        coherence_model_lda = CoherenceModel(model=lda_model, 
                                             texts=data_processed, 
                                             dictionary=id2word, 
                                             coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        
        print(f"    K={k}, Coherence: {coherence_lda:.4f}")
        results.append({'Num_Topics': k, 'Coherence_Score': coherence_lda})
        models[k] = lda_model

    # Save Coherence Table
    df_results = pd.DataFrame(results)
    df_results.to_csv('output/coherence_scores.csv', index=False)
    print("Saved 'output/coherence_scores.csv'.")
    
    # Identify Best Model (Highest Coherence)
    best_result = max(results, key=lambda x: x['Coherence_Score'])
    best_k = best_result['Num_Topics']
    best_score = best_result['Coherence_Score']
    print(f"Best Model: K={best_k} with Coherence={best_score:.4f}")
    
    best_model = models[best_k]

    # 6. Visualizations
    
    # A. Word Clouds for Top Topics
    print("Generating Word Clouds...")
    topics = best_model.show_topics(formatted=False, num_topics=best_k, num_words=20)
    
    for topic_idx, words in topics:
        topic_words_dict = {word: prob for word, prob in words}
        
        wc = WordCloud(background_color='white', width=800, height=400)
        wc.generate_from_frequencies(topic_words_dict)
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Topic {topic_idx}")
        # Make clean filename
        plt.savefig(f"output/plots/topic_{topic_idx}_wordcloud.png", bbox_inches='tight')
        plt.close()
        
    print(f"Saved {best_k} wordcloud plots to 'output/plots/'.")

    # B. pyLDAvis Visualization
    print("Generating pyLDAvis visualization...")
    try:
        vis = pyLDAvis.gensim_models.prepare(best_model, corpus, id2word)
        pyLDAvis.save_html(vis, 'output/lda_visualization.html')
        print("Saved 'output/lda_visualization.html'.")
    except Exception as e:
        print(f"Failed to generate pyLDAvis: {e}")

    print("\nProcessing complete. Check the 'output' directory for results.")

if __name__ == "__main__":
    main()
