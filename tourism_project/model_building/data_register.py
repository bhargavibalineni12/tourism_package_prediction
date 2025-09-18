import os
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
from google.colab import userdata  # For Colab secrets

#  Load Hugging Face token securely from Colab secrets
hf_token = userdata.get("HF_TOKEN")
if not hf_token:
    raise ValueError(" Hugging Face token not found in Colab secrets. Please add it first!")

# Initialize API with token
api = HfApi(token=hf_token)

# Define repo details
repo_id = "Bhargavi329/tourism-package-prediction"   # Replace with your HF username/repo
repo_type = "dataset"

#  Ensure dataset repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f" Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Repo '{repo_id}' not found. Creating new repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f" Repo '{repo_id}' created.")

# Path to dataset file
dataset_file_path = "/content/tourism_project/data/tourism.csv"
if not os.path.exists(dataset_file_path):
    raise FileNotFoundError(f" Dataset file not found at {dataset_file_path}")

#  Upload dataset file to Hugging Face Hub
api.upload_file(
    path_or_fileobj=dataset_file_path,
    path_in_repo="tourism.csv",
    repo_id=repo_id,
    repo_type=repo_type
)

print(f"Dataset {dataset_file_path} uploaded successfully to {repo_id}.")
