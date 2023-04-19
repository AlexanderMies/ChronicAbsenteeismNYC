import pandas as pd
import numpy as np
import os


def clean_col_name(col_name):
    """
    Cleans column names by removing spaces and making them lowercase.
    Also changes '#' to 'num' and '%' to 'pct'.
    Also replaces '+' with 'and'.
    """
    return (
        col_name.lower()
        .replace(" ", "_")
        .replace("#", "num")
        .replace("%", "pct")
        .replace("+", "and")
    )


# Load in raw demographic data by district.
raw_demographics = pd.read_excel(
    "Data/Raw/demographic-snapshot-2017-18-to-2021-22.xlsx",
    sheet_name="District",
)

# Clean column names.
raw_demographics.columns = [
    clean_col_name(col) for col in raw_demographics.columns
]

# Clean year column to match test year columns.
clean_year = lambda x: int("20" + x[-2:])
raw_demographics["year"] = raw_demographics["year"].apply(clean_year)

# Rename district column to match other data.
raw_demographics.rename(
    columns={"administrative_district": "district"}, inplace=True
)

# Save a clean version of the demographic data.
raw_demographics.to_csv("Data/Processed/demographic_data.csv", index=False)

# Drop the number of students per grade columns.
raw_demographics.drop(
    columns=[col for col in raw_demographics.columns if "grade_" in col],
    inplace=True,
)

# Load in data with attendance and test scores.
attendance_and_tests = pd.read_csv(
    "Data/Processed/merged_attendance_test_data.csv"
)

# Filter only to category == "All Students" and grade == "All Grades".
attendance_and_tests = attendance_and_tests.query(
    "category == 'All Students' and grade == 'All Grades'"
)

# Merge attendance and test data with demographic data.
merged_data = attendance_and_tests.merge(
    raw_demographics, on=["district", "year"]
)

assert (
    merged_data.shape[0] == attendance_and_tests.shape[0]
), "Merged data should have same number of rows as attendance and test data."

# Save merged data.
merged_data.to_csv(
    "Data/Processed/absentee_tests_demographics_all_grades.csv", index=False
)
