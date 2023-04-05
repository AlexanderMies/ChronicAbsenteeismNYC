import pandas as pd
import numpy as np
import os

os.getcwd()
os.listdir("Data/Raw")


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


# Load in attendance data
attendance_data = pd.read_excel(
    "Data/Raw/public-district-attendance-results-2018-2022.xlsx",
    sheet_name=None,
)
# Concatenate all versions of data.
# "category" column is the split variable.
attendance_cuts = pd.concat(
    [
        attendance_data[split]
        for split in ["All Students", "Ethnicity", "Gender"]
    ],
    axis=0,
)
# Clean column names.
attendance_cuts.columns = [
    clean_col_name(col) for col in attendance_cuts.columns
]
# Clean year column to match test year columns.
clean_year = lambda x: int("20" + x[-2:])
attendance_cuts["year"] = attendance_cuts["year"].apply(clean_year)

# Save attendance data.
attendance_cuts.to_csv("Data/Processed/attendance_data.csv", index=False)

# Load in test score data.
# Concatenate all versions of data. "category" column is the split variable.
math_test_data = pd.read_excel(
    "Data/Raw/district-math-results-2013-2022.xlsx",
    sheet_name=None,
)
math_test_cuts = pd.concat(
    [math_test_data[split] for split in ["All", "Ethnicity", "Gender"]],
    axis=0,
)
# Clean column names.
math_test_cuts.columns = [
    clean_col_name(col) for col in math_test_cuts.columns
]

ela_test_data = pd.read_excel(
    "Data/Raw/district-ela-results-2013-2022.xlsx",
    sheet_name=None,
)
ela_test_cuts = pd.concat(
    [ela_test_data[split] for split in ["All", "Ethnicity", "Gender"]],
    axis=0,
)
# Clean column names.
ela_test_cuts.columns = [clean_col_name(col) for col in ela_test_cuts.columns]

# Merge test data together.
test_data = math_test_cuts.merge(
    ela_test_cuts,
    on=["district", "grade", "year", "category"],
    suffixes=("_math", "_ela"),
    how="inner",
)
test_data.to_csv("Data/Processed/merged_ela_math_test_data.csv", index=False)

# Merge attendance data with test data.
attendance_test_data = attendance_cuts.merge(
    test_data, on=["district", "grade", "year", "category"], how="inner"
)

# Save merged attendance and test data.
# Note that this will only have 2018, 2019, and 2022 data.
attendance_test_data.to_csv(
    "Data/Processed/merged_attendance_test_data.csv", index=False
)
