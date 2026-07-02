# main.py

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def run_command(command):
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as error:
        print("Command failed:", error)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProcess stopped by user.")
        sys.exit(0)


def run_training():
    print("Starting model training...")
    run_command([sys.executable, "src/train_model.py"])


def run_evaluation():
    print("Starting model evaluation...")
    run_command([sys.executable, "src/evaluate_model.py"])


def run_prediction():
    print("Running sample prediction...")
    run_command([sys.executable, "src/predict.py"])


def run_streamlit():
    print("Starting Streamlit app...")
    run_command([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/streamlit_app.py"
    ])


def run_fastapi():
    print("Starting FastAPI server...")
    run_command([
        sys.executable,
        "-m",
        "uvicorn",
        "api.fastapi_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload"
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Customer Churn Prediction Project Runner"
    )

    parser.add_argument(
        "command",
        choices=[
            "train",
            "evaluate",
            "predict",
            "streamlit",
            "api"
        ],
        help="Choose what you want to run"
    )

    args = parser.parse_args()

    if args.command == "train":
        run_training()

    elif args.command == "evaluate":
        run_evaluation()

    elif args.command == "predict":
        run_prediction()

    elif args.command == "streamlit":
        run_streamlit()

    elif args.command == "api":
        run_fastapi()


if __name__ == "__main__":
    main()
