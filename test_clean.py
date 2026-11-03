import pandas as pd
from src.shipment_prep import clean_shipments

raw = pd.read_csv("data/raw_shipments.csv")
clean, medians = clean_shipments(raw)
print(len(clean))
print(medians)
print(clean["service_type"].unique())
print(clean.isna().sum())