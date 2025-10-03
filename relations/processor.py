import pandas as pd

class ArgumentDataProcessor:
    """
    Nettoyage minimal comme pendant l'entraînement
    """
    def clean_text(self, text):
        if pd.isna(text) or text is None:
            return ""
        return str(text).strip()