"""Optional fire model downloader.

The HSV detector is model-free. Place optional weights under ai/models/ using
this script only when a vetted model URL is configured for your deployment.
"""
from pathlib import Path

def main() -> None:
    target = Path(__file__).resolve().parents[2] / "models"
    target.mkdir(exist_ok=True)
    print(f"No mandatory fire weights; optional models belong in {target}")

if __name__ == "__main__":
    main()
