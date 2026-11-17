"""Print the model destination for an optional YAMNet ONNX download."""
from pathlib import Path

def main() -> None:
    target = Path(__file__).resolve().parents[2] / "models"
    target.mkdir(exist_ok=True)
    print(f"Download a vetted YAMNet ONNX file to {target / 'yamnet.onnx'}")

if __name__ == "__main__":
    main()
