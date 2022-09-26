import csv
import logging
import os
import pickle as pkl
from datetime import datetime as dt

def to_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        if isinstance(row, list):
            keys = set()
            for row_ in row:
                keys.update(row_.keys())
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            for row_ in row:
                writer.writerow(row_)
        else:
            writer = csv.DictWriter(f, fieldnames=sorted(list(row.keys())))
            writer.writeheader()
            writer.writerow(row)

def append_csv(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(list(row.keys())))
        if write_header: 
            writer.writeheader()
        writer.writerow(row)
