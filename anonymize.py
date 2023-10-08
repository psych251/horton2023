"""
Script for anonymizing data from Qualtrics
"""
import os
import csv
import pandas as pd
from argparse import ArgumentParser

cols_to_drop = [
    "RecipientLastName",
    "RecipientFirstName",
    "RecipientEmail",
    "ExternalDataReference",
    "IPAddress",
    "LocationLatitude",
    "LocationLongitude",
    "PROLIFIC_PID",
]


def anonymize_data_file(file_path):
    """
    Drop all identifying information from a data file.
    This would be way easier with pandas, but that would add a dependency.
    """
    indices_to_drop = []
    with open(file_path, "r") as f_in:
        reader = csv.reader(f_in)
        for header in reader:
            break
        for i, colname in enumerate(header):
            if colname in cols_to_drop:
                indices_to_drop.append(i)

        anon_file_path = file_path.replace(".csv", "-anon.csv")
        with open(anon_file_path, "w") as f_out:
            f_in.seek(0)
            for row in reader:
                cols_to_keep = [
                    row[i] for i in range(len(row)) if i not in indices_to_drop
                ]
                f_out.write(",".join(cols_to_keep) + "\n")


def get_raw_data_files(root_dir):
    """
    Retrieve data files from a directory.
    """
    all_data_files = []
    for path in os.walk(root_dir):
        # if the path contains a non-anonymized csv
        for filename in path[2]:
            if (
                filename.split(".")[-1] == "csv"
                and filename.split("-")[-1] != "anon.csv"
            ):
                all_data_files.append("/".join([path[0], filename]))

    return all_data_files


parser = ArgumentParser()
parser.add_argument("--root_dir", type=str, default="data")

if __name__ == "__main__":
    args = parser.parse_args()
    raw_data_files = get_raw_data_files(args.root_dir)

    for data_file in raw_data_files:
        print(f"anonymizing {data_file}...")
        anonymize_data_file(data_file)
