from pathlib import Path as _Path
_ROOT = str(_Path(__file__).resolve().parent)

import re
import numpy as np

# The core scoring logic (Strings -> Float)
def score_abc_text(text):
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    targets = ['A', 'B', 'C']
    hits = 0
    for i in range(min(3, len(sentences))):
        clean_sent = re.sub(r'^[^a-zA-Z]+', '', sentences[i])
        if clean_sent and clean_sent[0].upper() == targets[i]:
            hits += 1
    return hits / 3.0

# The Wrapper (Token IDs -> Float)
class LLaDAReward:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, token_ids):
        # Ensure we work with numpy for shape checking
        token_ids_np = np.array(token_ids)
        
        # 1. BATCH MODE (2D Input)
        if token_ids_np.ndim == 2:
            # tokenizer.batch_decode handles the loop C++ side (faster)
            # We convert to list because some tokenizers dislike numpy inputs directly
            texts = self.tokenizer.batch_decode(token_ids_np.tolist(), skip_special_tokens=True)
            
            # Vectorize the scoring
            scores = [score_abc_text(t) for t in texts]
            return np.array(scores)
            
        # 2. SINGLE MODE (1D Input)
        else:
            text = self.tokenizer.decode(token_ids_np.tolist(), skip_special_tokens=True)
            return score_abc_text(text)
