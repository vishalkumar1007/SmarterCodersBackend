from sentence_transformers import SentenceTransformer
import nltk
from typing import List, Tuple
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
nltk.data.path.append("C:\\nltk_data")
nltk.download('punkt', download_dir="C:\\nltk_data")

def tokenize_and_embed(content: str) -> Tuple[List[str], np.ndarray]:
    # Split into sentences first
    sentences = nltk.sent_tokenize(content)
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    for sentence in sentences:
        # Tokenize sentence
        tokens = nltk.word_tokenize(sentence)
        
        if current_tokens + len(tokens) > 500:
            # If current chunk would exceed 500 tokens, save it and start new chunk
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = len(tokens)
        else:
            # Add sentence to current chunk
            current_chunk += " " + sentence
            current_tokens += len(tokens)
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Generate embeddings
    embeddings = model.encode(chunks)
    
    print(f"Created {len(chunks)} chunks with average length: {sum(len(nltk.word_tokenize(chunk)) for chunk in chunks)/len(chunks):.0f} tokens")
    
    return chunks, embeddings