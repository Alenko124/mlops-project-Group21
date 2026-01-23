"""Data drift detection using Evidently framework for satellite image classification."""

import json
from typing import Optional

import pandas as pd
from google.cloud import storage
from loguru import logger

from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset, ClassificationPreset


class DriftDetector:
    """Detects data drift in satellite image predictions using Evidently."""

    def __init__(
        self,
        bucket_name: str = "mlops-group21",
        reference_folder: str = "predictions/reference_data",
        current_folder: str = "predictions/data_logs",
    ):
        """Initialize drift detector.

        Args:
            bucket_name: GCS bucket name
            reference_folder: Folder containing reference baseline data
            current_folder: Folder containing current production data
        """
        self.bucket_name = bucket_name
        self.reference_folder = reference_folder
        self.current_folder = current_folder
        self.client = storage.Client()
        logger.info(f"Initialized DriftDetector for bucket: {bucket_name}")

    def load_gcs_json_files(self, folder: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load JSON files from GCS folder into DataFrame.

        Args:
            folder: GCS folder path (e.g., 'predictions/data_logs')
            limit: Maximum number of files to load (None for all)

        Returns:
            DataFrame with loaded records
        """
        try:
            bucket = self.client.bucket(self.bucket_name)
            blobs = bucket.list_blobs(prefix=folder)

            records = []
            for i, blob in enumerate(blobs):
                if limit and i >= limit:
                    break

                try:
                    content = blob.download_as_string()
                    record = json.loads(content)
                    records.append(record)
                except Exception as e:
                    logger.warning(f"Failed to load {blob.name}: {e}")
                    continue

            if not records:
                logger.warning(f"No records found in {folder}")
                return pd.DataFrame()

            return pd.DataFrame(records)

        except Exception as e:
            logger.error(f"Failed to load data from {folder}: {e}")
            return pd.DataFrame()

    # def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
    #     """Prepare DataFrame by flattening image features.

    #     Args:
    #         df: Raw DataFrame from GCS

    #     Returns:
    #         Processed DataFrame with flattened image features
    #     """
    #     if df.empty:
    #         return df

    #     # Flatten image_features dict into separate columns
    #     if "image_features" in df.columns:
    #         features_df = pd.json_normalize(df["image_features"])
    #         df = pd.concat([df, features_df], axis=1)
    #         df = df.drop(columns=["image_features"])

    #     # Convert timestamp to datetime
    #     if "timestamp" in df.columns:
    #         df["timestamp"] = pd.to_datetime(df["timestamp"])

    #     # Remove true_label if it's all null (production data doesn't have labels)
    #     if "true_label" in df.columns and df["true_label"].isna().all():
    #         df = df.drop(columns=["true_label"])

    #     return df

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        # Flatten image_features
        if "image_features" in df.columns:
            features_df = pd.json_normalize(df["image_features"])
            df = pd.concat([df.drop(columns=["image_features"]), features_df], axis=1)

        # Ensure expected columns exist in BOTH ref and current
        expected_cols = [
            "timestamp",
            "predicted_class",
            "predicted_class_name",
            "confidence",
            "true_label",
            "brightness",
            "contrast",
            "sharpness",
        ]
        for c in expected_cols:
            if c not in df.columns:
                df[c] = pd.NA

        # Types
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

        for c in ["confidence", "brightness", "contrast", "sharpness"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # Treat predicted_class as categorical for drift (optional but recommended)
        df["predicted_class"] = df["predicted_class"].astype("Int64")
        df["predicted_class_name"] = df["predicted_class_name"].astype("string")
        df["true_label"] = df["true_label"].astype("Int64")

        return df


    def generate_drift_report(self, num_current_records: Optional[int] = None) -> str:
        """Generate HTML drift report comparing reference and current data.

        Args:
            num_current_records: Number of recent records to analyze (None for all)

        Returns:
            HTML content of the report
        """
        try:
            logger.info("Loading reference data...")
            reference_df = self.load_gcs_json_files(self.reference_folder)

            logger.info(f"Loading current data (limit: {num_current_records})...")
            current_df = self.load_gcs_json_files(self.current_folder, limit=num_current_records)

            if reference_df.empty:
                logger.error("Reference data is empty")
                return "<h1>Error: Reference data not available</h1>"

            if current_df.empty:
                logger.error("Current data is empty")
                return "<h1>Error: Current production data not available</h1>"

            # # Prepare dataframes
            # reference_df = self.prepare_dataframe(reference_df)
            # current_df = self.prepare_dataframe(current_df)

            # logger.info(f"Reference data shape: {reference_df.shape}, " f"Current data shape: {current_df.shape}")

            # # Build metrics list - only add ClassificationPreset if true_label is available
            # metrics = [DataDriftPreset()]

            # if "true_label" in reference_df.columns and "true_label" in current_df.columns:
            #     if not reference_df["true_label"].isna().all() and not current_df["true_label"].isna().all():
            #         metrics.append(ClassificationPreset())
            #         logger.info("Added ClassificationPreset to report")
            # else:
            #     logger.info("true_label not available in datasets, using DataDriftPreset only")

            # # Create report with available metrics
            # report = Report(metrics=metrics)

            # report.run(
            #     reference_data=reference_df,
            #     current_data=current_df,
            # )

            # return report.get_html()

            # Prepare

            reference_df = self.prepare_dataframe(reference_df)
            current_df = self.prepare_dataframe(current_df)

            # Drift columns you actually want
            drift_cols = [
                "brightness",
                "contrast",
                "sharpness",
                "confidence",
                "predicted_class_name",  # class distribution drift
            ]

            # Ensure they exist in both
            drift_cols = [c for c in drift_cols if c in reference_df.columns and c in current_df.columns]
            if not drift_cols:
                return "<h1>Error: No common drift columns available</h1>"

            report = Report(metrics=[DataDriftPreset(columns=drift_cols)])
            report.run(
                reference_data=reference_df[drift_cols],
                current_data=current_df[drift_cols],
            )
            return report.get_html()


        except Exception as e:
            logger.error(f"Failed to generate drift report: {e}")
            return f"<h1>Error generating report: {str(e)}</h1>"

    def generate_feature_drift_report(self, num_current_records: Optional[int] = None) -> str:
        """Generate focused HTML report on image feature drift.

        Args:
            num_current_records: Number of recent records to analyze (None for all)

        Returns:
            HTML content of the report
        """
        try:
            logger.info("Loading reference data for feature analysis...")
            reference_df = self.load_gcs_json_files(self.reference_folder)

            logger.info(f"Loading current data (limit: {num_current_records})...")
            current_df = self.load_gcs_json_files(self.current_folder, limit=num_current_records)

            if reference_df.empty or current_df.empty:
                return "<h1>Error: Insufficient data for analysis</h1>"

            # Prepare dataframes
            reference_df = self.prepare_dataframe(reference_df)
            current_df = self.prepare_dataframe(current_df)

            # Select only numeric feature columns for drift analysis
            feature_columns = ["brightness", "contrast", "sharpness", "confidence"]
            available_features = [col for col in feature_columns if col in reference_df.columns and col in current_df.columns]

            if not available_features:
                return "<h1>Error: No feature columns found</h1>"

            # Create report focused on numeric features
            report = Report(
                metrics=[
                    DataDriftPreset(columns=available_features),
                ]
            )

            report.run(
                reference_data=reference_df[available_features],
                current_data=current_df[available_features],
            )

            return report.get_html()

        except Exception as e:
            logger.error(f"Failed to generate feature drift report: {e}")
            return f"<h1>Error generating report: {str(e)}</h1>"

    def get_drift_summary(self, num_current_records: int = 100) -> dict:
        """Get summary statistics about detected drifts.

        Args:
            num_current_records: Number of recent records to analyze

        Returns:
            Dictionary with drift summary
        """
        try:
            reference_df = self.load_gcs_json_files(self.reference_folder)
            current_df = self.load_gcs_json_files(self.current_folder, limit=num_current_records)

            if reference_df.empty or current_df.empty:
                return {"error": "Insufficient data"}

            reference_df = self.prepare_dataframe(reference_df)
            current_df = self.prepare_dataframe(current_df)

            summary = {
                "reference_records": len(reference_df),
                "current_records": len(current_df),
                "reference_stats": {},
                "current_stats": {},
            }

            # Compute statistics for numeric columns
            numeric_cols = ["brightness", "contrast", "sharpness", "confidence"]
            for col in numeric_cols:
                if col in reference_df.columns:
                    summary["reference_stats"][col] = {
                        "mean": float(reference_df[col].mean()),
                        "std": float(reference_df[col].std()),
                        "min": float(reference_df[col].min()),
                        "max": float(reference_df[col].max()),
                    }

                if col in current_df.columns:
                    summary["current_stats"][col] = {
                        "mean": float(current_df[col].mean()),
                        "std": float(current_df[col].std()),
                        "min": float(current_df[col].min()),
                        "max": float(current_df[col].max()),
                    }

            # Class distribution
            if "predicted_class_name" in reference_df.columns:
                summary["reference_class_dist"] = reference_df["predicted_class_name"].value_counts().to_dict()
            if "predicted_class_name" in current_df.columns:
                summary["current_class_dist"] = current_df["predicted_class_name"].value_counts().to_dict()

            return summary

        except Exception as e:
            logger.error(f"Failed to get drift summary: {e}")
            return {"error": str(e)}
