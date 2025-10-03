import numpy as np

def generate_embeddings(arg1, arg2, embedding_model, processor):
    arg1_clean = processor.clean_text(arg1)
    arg2_clean = processor.clean_text(arg2)

    parent_emb = embedding_model.encode([arg1_clean])[0]
    child_emb = embedding_model.encode([arg2_clean])[0]

    diff_emb = np.abs(parent_emb - child_emb)
    product_emb = parent_emb * child_emb
    cos_sim = np.array([
        np.dot(parent_emb, child_emb) /
        ((np.linalg.norm(parent_emb) + 1e-8) * (np.linalg.norm(child_emb) + 1e-8))
    ])

    return np.concatenate([parent_emb, child_emb, diff_emb, product_emb, cos_sim])
