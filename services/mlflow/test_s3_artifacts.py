#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "mlflow",
#     "boto3",
# ]
# ///
"""
Quick test script to verify MLflow S3 artifact logging.
This script creates a test run and logs artifacts to verify S3 connectivity.

NOTE: S3 credentials are required locally because the MLflow client directly uploads
artifacts to S3 (not through the MLflow server). This is by design for performance.

Usage:
    # Set S3 credentials from your .env file
    export AWS_ACCESS_KEY_ID=your-access-key
    export AWS_SECRET_ACCESS_KEY=your-secret-key
    export MLFLOW_S3_ENDPOINT_URL=https://your-s3-endpoint.example.com

    # Run the test
    uv run services/mlflow/test_s3_artifacts.py

    # Or inline:
    AWS_ACCESS_KEY_ID=key AWS_SECRET_ACCESS_KEY=secret uv run services/mlflow/test_s3_artifacts.py
"""

import mlflow
import os
import tempfile
from datetime import datetime


def test_s3_artifact_logging():
    """Test artifact logging to S3 via MLflow."""

    # Get MLflow tracking URI from environment or use default
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Set up MLflow authentication
    username = os.getenv("MLFLOW_TRACKING_USERNAME", "data_mining")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD", "admin123_datadays")

    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    # Set up S3 credentials (required for client-side artifact upload)
    # Check if S3 credentials are set
    aws_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")

    if not aws_key or not aws_secret:
        print("\n⚠️  WARNING: AWS credentials not found!")
        print("Set these environment variables:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - MLFLOW_S3_ENDPOINT_URL (optional, for custom S3 endpoints)")
        print("\nExample:")
        print("  export AWS_ACCESS_KEY_ID=your-key")
        print("  export AWS_SECRET_ACCESS_KEY=your-secret")
        raise ValueError("AWS credentials required for S3 artifact logging")

    mlflow.set_tracking_uri(tracking_uri)

    print(f"MLflow Tracking URI: {tracking_uri}")
    print(f"MLflow Auth: {username}")
    print(f"S3 Endpoint: {s3_endpoint or 'default (AWS S3)'}")
    print(f"AWS Access Key: {aws_key[:8]}..." if aws_key else "Not set")
    print(f"\nTesting S3 artifact logging...")

    # Set experiment
    experiment_name = "s3_connection_test"
    mlflow.set_experiment(experiment_name)

    # Start a test run
    with mlflow.start_run(
        run_name=f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ):
        # Log parameters
        mlflow.log_param("test_param", "test_value")
        mlflow.log_param("timestamp", datetime.now().isoformat())

        # Log metrics
        mlflow.log_metric("test_metric", 42.0)
        mlflow.log_metric("accuracy", 0.95)

        # Create and log a test text artifact
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is a test artifact to verify S3 connectivity.\n")
            f.write(f"Created at: {datetime.now().isoformat()}\n")
            f.write("If you can read this, S3 artifact logging is working!\n")
            temp_file = f.name

        try:
            mlflow.log_artifact(temp_file, "test_artifacts")
            print("✓ Successfully logged text artifact")
        finally:
            os.unlink(temp_file)

        # Create and log a test JSON artifact
        import json

        test_data = {
            "test": "s3_connection",
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "data": [1, 2, 3, 4, 5],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f, indent=2)
            temp_json = f.name

        try:
            mlflow.log_artifact(temp_json, "test_artifacts")
            print("✓ Successfully logged JSON artifact")
        finally:
            os.unlink(temp_json)

        # Log a dictionary as artifacts
        artifact_dict = {
            "model_config.txt": "test configuration",
            "results.txt": "test results",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            for filename, content in artifact_dict.items():
                filepath = os.path.join(tmpdir, filename)
                with open(filepath, "w") as f:
                    f.write(content)

            mlflow.log_artifacts(tmpdir, "test_artifacts/batch")
            print("✓ Successfully logged multiple artifacts")

        # Get run info
        run = mlflow.active_run()
        print(f"\n✓ Test run completed successfully!")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Experiment ID: {run.info.experiment_id}")
        print(f"  Artifact URI: {run.info.artifact_uri}")

        return run.info


if __name__ == "__main__":
    try:
        run_info = test_s3_artifact_logging()
        print("\n✓ S3 artifact logging test PASSED")
        print(f"\nYou can verify the artifacts at: {run_info.artifact_uri}")
    except Exception as e:
        print(f"\n✗ S3 artifact logging test FAILED")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
