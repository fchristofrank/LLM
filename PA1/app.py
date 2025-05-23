import streamlit as st
import nltk
import string
import pickle
import os
from collections import Counter
from nltk.corpus import reuters, brown, gutenberg, webtext
import logging
import time

# Configure page
st.set_page_config(page_title="Word Completion", page_icon="📝")
st.title("Word Completion")
st.write("Enter a prefix and get word completions based on n-gram language model")

# Cache file base name
CACHE_FILE_BASE = "optimized_ngram_model_"

# Disable NLTK download messages
logging.getLogger("nltk").setLevel(logging.ERROR)

# Constants for optimization
MAX_VOCAB_SIZE = 1000000  # Limit vocabulary to top 50,000 words
MAX_TOKENS = 1000000    # Limit to 1 million tokens for training

def preprocess_text(text):
    """Clean and normalize text"""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ''.join([char for char in text if not char.isdigit()])
    text = ' '.join(text.split())
    return text

@st.cache_resource
def get_training_data():
    """Get raw text data from multiple corpora"""
    # Download required corpora silently
    nltk.download('reuters', quiet=True)
    nltk.download('brown', quiet=True)
    nltk.download('gutenberg', quiet=True)
    nltk.download('webtext', quiet=True)
    nltk.download('punkt', quiet=True)
    
    # Load Reuters corpus
    reuters_text = ' '.join(reuters.raw(file_id) for file_id in reuters.fileids())
    
    # Load Brown corpus
    brown_text = ' '.join(brown.raw(file_id) for file_id in brown.fileids())
    
    # Load Gutenberg corpus
    gutenberg_text = ' '.join(gutenberg.raw(file_id) for file_id in gutenberg.fileids())
    
    # Load Webtext corpus
    webtext_text = ' '.join(webtext.raw(file_id) for file_id in webtext.fileids())
    
    # Create dataset
    combined_text = (
        reuters_text + " " + brown_text + " " + webtext_text
    )
    
    # Preprocess the combined text
    combined_text = preprocess_text(combined_text)
    
    return combined_text

def get_limited_training_data(combined_text, max_tokens=MAX_TOKENS, max_vocab=MAX_VOCAB_SIZE):
    """Get limited training data for better performance"""
    # Split into tokens
    all_tokens = combined_text.split()
    
    # Limit to max_tokens for faster processing
    if len(all_tokens) > max_tokens:
        all_tokens = all_tokens[:max_tokens]
    
    # Get word frequencies
    word_counts = Counter(all_tokens)
    
    # Limit vocabulary to top max_vocab words
    limited_vocab = set([word for word, _ in word_counts.most_common(max_vocab)])
    
    # Filter tokens to keep only those in our limited vocabulary
    filtered_tokens = [token if token in limited_vocab else '<UNK>' for token in all_tokens]
    
    # Add <UNK> token to represent unknown words
    limited_vocab.add('<UNK>')
    
    return filtered_tokens, list(limited_vocab)

def create_ngrams(tokens, n):
    """Create n-grams from tokens"""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    return ngrams

def count_ngrams(ngrams):
    """Count frequency of each n-gram"""
    return Counter(ngrams)

def calculate_probability(word, context, ngram_counts, context_counts, vocabulary_size, smoothing='kneser_ney', alpha=0.1):
    """Calculate probability with various smoothing strategies"""
    ngram = context + (word,)
    
    if smoothing == 'none':
        # Maximum Likelihood Estimation (no smoothing)
        if context not in context_counts or ngram not in ngram_counts:
            return 0.0
        return ngram_counts.get(ngram, 0) / context_counts.get(context, 1)
    
    elif smoothing == 'laplace':
        # Laplace (Add-alpha) Smoothing
        numerator = ngram_counts.get(ngram, 0) + alpha
        denominator = context_counts.get(context, 0) + (alpha * vocabulary_size)
        return numerator / denominator
    
    elif smoothing == 'kneser_ney':
        # Simplified Kneser-Ney smoothing
        d = 0.75 if alpha is None else alpha
        count = ngram_counts.get(ngram, 0)
        
        if count > 0:
            discounted_count = max(0, count - d) / context_counts.get(context, 1)
        else:
            discounted_count = 0
            
        unique_continuations = sum(1 for ng in ngram_counts if ng[:-1] == context)
        backoff_weight = d * unique_continuations / context_counts.get(context, 1) if unique_continuations > 0 else 0.1
        
        # Use a simpler continuation probability for speed
        continuation_prob = 1/vocabulary_size
        
        return discounted_count + (backoff_weight * continuation_prob)
    
    elif smoothing == 'interpolation':
        # Linear interpolation of different n-gram orders
        lambda1 = 0.7  # Weight for the full n-gram
        lambda2 = 0.2  # Weight for the (n-1)-gram
        lambda3 = 0.1  # Weight for the unigram
        
        # Full n-gram probability
        if context not in context_counts:
            p_ngram = 0
        else:
            p_ngram = ngram_counts.get(ngram, 0) / context_counts.get(context, 1)
        
        # Simplified backoff for speed
        p_unigram = 1/vocabulary_size
        
        # Combine probabilities
        return (lambda1 * p_ngram) + (lambda3 * p_unigram)
    
    else:
        # Default to simple Laplace smoothing
        numerator = ngram_counts.get(ngram, 0) + 0.1
        denominator = context_counts.get(context, 0) + (0.1 * vocabulary_size)
        return numerator / denominator

def predict_next_word(context, vocabulary, ngram_counts, context_counts, n, smoothing='kneser_ney', alpha=0.1, top_k=5):
    """Predict most likely next words given context"""
    # Ensure context is right length
    context = tuple(context[-(n-1):]) if len(context) >= n-1 else context
    
    # For efficiency, only check words that have appeared after this context
    candidates = {}
    
    # Only check exact matches (for speed)
    for ngram in ngram_counts:
        if len(ngram) == n and ngram[:-1] == context:
            word = ngram[-1]
            # Use cached counts for faster computation
            count = ngram_counts[ngram]
            context_count = context_counts[context]
            if context_count > 0:  # Avoid division by zero
                prob = count / context_count
                candidates[word] = prob
    
    # Apply smoothing to the top candidates (for speed)
    if candidates:
        # Sort by raw count and take top 20 for smoothing
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)[:20]
        smoothed_candidates = {}
        
        for word, _ in sorted_candidates:
            prob = calculate_probability(
                word, context, ngram_counts, context_counts, len(vocabulary), 
                smoothing=smoothing, alpha=alpha
            )
            smoothed_candidates[word] = prob
        
        return sorted(smoothed_candidates.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # If no candidates found, return most common words
    else:
        # Use pre-computed common words
        common_words = ['the', 'of', 'and', 'to', 'a', 'in', 'that', 'is', 'was', 'he', 
                        'for', 'it', 'with', 'as', 'his', 'on', 'be', 'at', 'by', 'i']
        return [(word, 0.001) for word in common_words[:top_k]]

def generate_completion(prefix_tokens, vocabulary, ngram_counts, context_counts, n, smoothing='kneser_ney', alpha=0.1, max_length=5):
    """Generate text completion starting with prefix"""
    result = list(prefix_tokens)
    
    # Replace unknown words with <UNK> token
    for i, token in enumerate(result):
        if token not in vocabulary:
            result[i] = '<UNK>'
    
    for _ in range(max_length):
        # Get context from the last n-1 tokens
        if len(result) < n-1:
            context = tuple(result)
        else:
            context = tuple(result[-(n-1):])
        
        # Predict next words
        predictions = predict_next_word(
            context, vocabulary, ngram_counts, context_counts, n,
            smoothing=smoothing, alpha=alpha
        )
        
        # Select next word
        if not predictions:
            next_word = 'the'  # Default to common word
        else:
            # Use most likely word for better quality
            next_word = predictions[0][0]
        
        # Skip UNK in output
        if next_word == '<UNK>':
            next_word = 'the'
            
        result.append(next_word)
    
    return ' '.join(result)

def train_model(n=3, smoothing='kneser_ney', alpha=0.1):
    """Train n-gram model with specified parameters"""
    start_time = time.time()
    
    # Create cache file name based on parameters
    cache_file = f"{CACHE_FILE_BASE}n{n}_{smoothing}_a{alpha}.pkl"
    
    # Check if cached model exists
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                model_data = pickle.load(f)
                # Verify this is the right model
                if (model_data.get('n') == n and 
                    model_data.get('smoothing') == smoothing and 
                    model_data.get('alpha') == alpha):
                    return model_data
        except Exception:
            # If loading fails, continue to train
            pass
    
    # Get the training data (cached by the decorator)
    combined_text = get_training_data()
    
    # Get limited training data for better performance
    train_tokens, vocabulary = get_limited_training_data(combined_text)
    
    # Create n-grams
    ngram_list = create_ngrams(train_tokens, n)
    ngram_counts = count_ngrams(ngram_list)
    
    # Create context (n-1)-grams
    context_list = create_ngrams(train_tokens, n-1)
    context_counts = count_ngrams(context_list)
    
    # Create model data
    model_data = {
        'vocabulary': vocabulary,
        'ngram_counts': ngram_counts,
        'context_counts': context_counts,
        'n': n,
        'smoothing': smoothing,
        'alpha': alpha,
        'token_count': len(train_tokens)
    }
    
    # Save to cache
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(model_data, f)
    except Exception:
        # Ignore saving errors
        pass
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training completed in {training_time:.2f} seconds")
        
    return model_data

# Main app logic
st.sidebar.title("Model Parameters")

# N-gram size selection
n_value = st.sidebar.selectbox(
    "N-gram Size",
    [2, 3, 4],  # Removed 5-grams for speed
    index=1,
    help="Higher values use more context but need more data. 3 is usually best."
)

# Smoothing method selection
smoothing_method = st.sidebar.selectbox(
    "Smoothing Method",
    ["kneser_ney", "laplace", "none"],  # Removed interpolation for speed
    index=0,
    help="Kneser-Ney generally performs best for most text."
)

# Alpha value for smoothing
alpha_value = st.sidebar.slider(
    "Smoothing Parameter (α)",
    0.01, 1.0, 0.1, 0.1,  # Larger step for fewer options
    help="Controls smoothing strength. Lower for less smoothing, higher for more."
)

# Force retrain option
retrain = st.sidebar.button("Retrain Model")

# Processing message placeholder
processing_message = st.empty()

# Train model with specified parameters
if 'model_data' not in st.session_state or retrain or (
    st.session_state.get('model_data', {}).get('n') != n_value or
    st.session_state.get('model_data', {}).get('smoothing') != smoothing_method or
    st.session_state.get('model_data', {}).get('alpha') != alpha_value
):
    processing_message.info("Processing... Please wait.")
    st.session_state.model_data = train_model(
        n=n_value,
        smoothing=smoothing_method,
        alpha=alpha_value
    )
    processing_message.empty()

# Word completion section
st.subheader("Enter a prefix:")
prefix = st.text_input("", value="what can be")
max_words = st.slider("Number of words to generate:", 1, 10, 5)  # Reduced max to 10

# Generate completions
generate_button = st.button("Complete Text")
if generate_button:
    if prefix:
        with st.spinner("Generating..."):
            start_time = time.time()
            
            # Process prefix
            prefix_clean = preprocess_text(prefix)
            prefix_tokens = prefix_clean.split()
            
            # Generate completion
            completion = generate_completion(
                prefix_tokens=prefix_tokens,
                vocabulary=st.session_state.model_data['vocabulary'],
                ngram_counts=st.session_state.model_data['ngram_counts'],
                context_counts=st.session_state.model_data['context_counts'],
                n=st.session_state.model_data['n'],
                smoothing=st.session_state.model_data['smoothing'],
                alpha=st.session_state.model_data['alpha'],
                max_length=max_words
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Display completion
            st.markdown("### Completion:")
            st.markdown(f"**{completion}**")
            st.caption(f"Generated in {generation_time:.2f} seconds")
            
            # Show next word predictions
            if prefix_tokens:
                n_value = st.session_state.model_data['n']
                context = tuple(prefix_tokens[-(n_value-1):]) if len(prefix_tokens) >= n_value-1 else tuple(prefix_tokens)
                
                predictions = predict_next_word(
                    context=context,
                    vocabulary=st.session_state.model_data['vocabulary'],
                    ngram_counts=st.session_state.model_data['ngram_counts'],
                    context_counts=st.session_state.model_data['context_counts'],
                    n=n_value,
                    smoothing=st.session_state.model_data['smoothing'],
                    alpha=st.session_state.model_data['alpha']
                )
                
                st.markdown("### Most likely next words:")
                cols = st.columns(min(5, len(predictions)))
                for i, (word, prob) in enumerate(predictions[:5]):
                    st.write(f"**{word}** — Probability: `{prob:.4f}`")
    else:
        st.error("Please enter a prefix.")


# Display model info in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Current Model")
st.sidebar.info(f"Tokens: {st.session_state.model_data['token_count']:,}")
st.sidebar.info(f"Vocabulary: {len(st.session_state.model_data['vocabulary']):,} words")
st.sidebar.info(f"Unique {n_value}-grams: {len(st.session_state.model_data['ngram_counts']):,}")

# Footer
st.markdown("---")
st.caption("Optimized Word Completion App | Built with Streamlit and NLTK")

# Minimal about section (collapsed by default)
with st.expander("About this app", expanded=False):
    st.markdown("""
    This app uses an optimized n-gram language model trained on a combined corpus of text. The vocabulary has been limited to the most common 50,000 words for better performance.
    
    **Tips for Best Results:**
    - Try n=3 (trigrams) with Kneser-Ney smoothing
    - Common phrases as prefixes generally give better results
    - If completions are too slow, lower the "Number of words to generate"
    """)