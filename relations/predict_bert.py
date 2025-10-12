import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_bert_model(model_path="../models/bert-argument", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    return model, tokenizer, device

def predict_relation(parent_text, child_text, model, tokenizer, device, max_length=256):
    """
    Predicts whether the relation between parent and child is Support or Attack.
    """
    model.eval()
    
    # Tokenization
    encoding = tokenizer(
        parent_text,
        child_text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation='only_second',
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    relation = "Support" if pred == 1 else "Attack"

    return {
        "predicted_label": relation,
        "probability": confidence,
        "confidence": confidence
    }
