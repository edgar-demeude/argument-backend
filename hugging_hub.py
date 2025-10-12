# File to automatically push the model to hugging face hub

from huggingface_hub import upload_folder

upload_folder(
    repo_id="edgar-demeude/bert-argument",
    folder_path="./models/bert-argument",
    repo_type="model"
)
