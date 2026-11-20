"""Training-only French caption retrieval with confidence-masked pose blending."""
from collections import Counter
import math

import numpy as np

from .dataset import tokens


def terms(text):
    words = tokens(text)
    counts = Counter(("word", word) for word in words)
    for word in words:
        padded = f"^{word}$"
        counts.update(("trigram", padded[i:i+3]) for i in range(len(padded)-2))
    return counts


class CaptionIndex:
    def __init__(self, captions):
        documents = [terms(caption) for caption in captions]
        frequencies = Counter(term for document in documents for term in document)
        self.idf = {term: math.log((1+len(documents))/(1+count))+1
                    for term, count in frequencies.items()}
        self.columns = {term: i for i, term in enumerate(self.idf)}
        self.vectors = np.stack([self.vector(document) for document in documents])

    def vector(self, counts):
        vector = np.zeros(len(self.idf), dtype=np.float32)
        for term, count in counts.items():
            if term in self.columns:
                vector[self.columns[term]] = (1+math.log(count))*self.idf[term]
        return vector/max(float(np.linalg.norm(vector)), 1e-12)

    def neighbors(self, caption, limit):
        similarities = self.vectors @ self.vector(terms(caption))
        ranked = np.argsort(-similarities, kind="stable")[:limit]
        return ranked[similarities[ranked] > 0]


def trajectory(train, indices, template, duration, prior_weight):
    if len(indices) == 0:
        return template, duration
    examples = [train[i] for i in indices]
    masks = np.stack([item[2] for item in examples])
    poses = np.stack([item[1] for item in examples])
    counts = masks.sum(axis=0)
    pooled = np.divide(np.where(masks[..., None], poses, 0).sum(axis=0),
                       counts.clip(min=1)[..., None])
    pooled = np.where(counts[..., None] > 0, pooled, template)
    seconds = np.mean([item[3] for item in examples])
    return (1-prior_weight)*pooled+prior_weight*template, float((1-prior_weight)*seconds+prior_weight*duration)
