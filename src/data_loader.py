import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from sklearn.preprocessing import StandardScaler


class CountryDataLoader:
    def __init__(
        self,
        data_path: Optional[Path] = None,
        standardize: bool = False,
        countries: Optional[List[str]] = None,
        mode: str = "train",  # new parameter: "train" or "predict"
    ):
        """
        Args:
            data_path: Root data folder path (overrides mode if provided).
            standardize: Whether to standardize each country's data.
            countries: List of countries to load (or None = load all).
            mode: Either "train" or "predict" to auto-select data folder.
        """
        base_path = Path(__file__).resolve().parent.parent

        if data_path:
            self.data_path = Path(data_path)
        else:
            self.data_path = base_path / ("Train" if mode == "train" else "Test")

        self.standardize = standardize
        self.requested_countries = countries
        self.pretrain_dict: Dict[str, pd.DataFrame] = {}
        self.train_dict: Dict[str, pd.DataFrame] = {}

        self.Z: Optional[np.ndarray] = None

    def load_data(self, split_date: Optional[str] = None) -> None:
        """
        Loads country-level data. Optionally splits each country dataset
        into pre-train and train sets based on a date column named 'eom'.

        Args:
            split_date: If provided, splits into pretrain (< date) and train (>= date).
        """
        self.data_dict = {}
        self.pretrain_dict = {}
        self.train_dict = {}

        for file in os.listdir(self.data_path):
            if file.endswith(".parquet"):
                country_code = file.split(".")[0]

                if self.requested_countries and country_code not in self.requested_countries:
                    continue

                df = pd.read_parquet(self.data_path / file)

                # Ensure datetime column exists and is datetime type
                if "eom" not in df.columns:
                    raise ValueError(f"Missing 'eom' column in {file}")
                df["eom"] = pd.to_datetime(df["eom"], format="%Y%m%d")

                # Standardize all non-date columns
                if self.standardize:
                    cols = df.columns.drop("eom")
                    df[cols] = StandardScaler().fit_transform(df[cols])

                if split_date:
                    split_ts = pd.to_datetime(split_date)
                    self.pretrain_dict[country_code] = df[df["eom"] < split_ts].reset_index(drop=True)
                    self.train_dict[country_code] = df[df["eom"] >= split_ts].reset_index(drop=True)
                else:
                    self.data_dict[country_code] = df.reset_index(drop=True)

        if split_date and not self.train_dict:
            raise ValueError("No training data found after the split date.")
        elif not split_date and not self.data_dict:
            raise ValueError("No data loaded.")


    def get_data(self) -> Dict[str, pd.DataFrame]:
        """Returns the dictionary of loaded country data."""
        return self.data_dict

    def available_countries(self) -> List[str]:
        """Returns a list of country codes available in the data directory."""
        return [
            file.split(".")[0]
            for file in os.listdir(self.data_path)
            if file.endswith(".parquet")
        ]
