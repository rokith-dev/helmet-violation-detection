
from huggingface_hub import snapshot_download

def main():
    path = snapshot_download(
        repo_id="thundarstrom/traffic-helmet-violation",
        repo_type="dataset",
        local_dir="data/external/helmet_combined",
    )

    print("Dataset download completed.")
    print(f"Dataset location: {path}")

if __name__ == "__main__":
    main()
