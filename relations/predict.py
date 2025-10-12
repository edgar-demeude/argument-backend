import torch
from .embeddings import generate_embeddings

def predict_relation_old(arg1, arg2, model, embedding_model, processor, best_threshold, label_encoder, model_type="pytorch"):
    embeddings = generate_embeddings(arg1, arg2, embedding_model, processor)

    if model_type == "pytorch":
        model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(embeddings).unsqueeze(0)
            prob = torch.sigmoid(model(tensor)).item()
        prediction = 1 if prob > best_threshold else 0
    else:
        prob = model.predict_proba(embeddings.reshape(1, -1))[0][1]
        prediction = 1 if prob > best_threshold else 0

    return {
        "predicted_label": label_encoder.inverse_transform([prediction])[0],
        "probability": prob,
        "confidence": abs(prob - 0.5) * 2
    }
