"""Load the UCI Online Retail II dataset (both sheets) into one ordered table."""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pandas as pd

from src.utils.config import ROOT, load_config, rpath
from src.utils.logging_setup import get_logger

log = get_logger(__name__)

COLUMNS = {
    "Invoice": "invoice",
    "StockCode": "stock_code",
    "Description": "description",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_date",
    "Price": "price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def _xlsx_to_frame(path: Path, sheets: list[str]) -> pd.DataFrame:
    """Stream the xlsx with openpyxl read-only mode (low memory) and return one frame."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as tmp:
        w = csv.writer(tmp)
        w.writerow(list(COLUMNS))
        for sheet in sheets:
            log.info("reading sheet=%s", sheet)
            for i, row in enumerate(wb[sheet].iter_rows(values_only=True)):
                if i:
                    w.writerow(row[: len(COLUMNS)])
        name = tmp.name
    return pd.read_csv(name, dtype={"Invoice": str, "StockCode": str, "Customer ID": str})


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    """Load both sheets, standardize names and types, sort chronologically."""
    cfg = cfg or load_config()
    path = rpath(cfg, "raw_xlsx")
    if path.exists():
        df = _xlsx_to_frame(path, cfg["data"]["sheets"])
    else:
        log.info("raw xlsx missing, fetching via ucimlrepo id=%s", cfg["data"]["uci_id"])
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=cfg["data"]["uci_id"])
        df = ds.data.original if ds.data.original is not None else ds.data.features
        df = df.rename(columns={"CustomerID": "Customer ID"})
        df = df[list(COLUMNS)].astype({"Invoice": str, "StockCode": str})
    df = df.rename(columns=COLUMNS)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.sort_values("invoice_date", kind="stable").reset_index(drop=True)
    log.info("loaded rows=%d", len(df))
    return df


if __name__ == "__main__":
    print(load_raw().head(), ROOT)  # noqa: T201 (manual smoke only)
