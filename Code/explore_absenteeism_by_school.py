import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load in attendance data
attendance = pd.read_csv("Data/Processed/attendance_data.csv").query(
    "category == 'All Students' and grade == 'All Grades'"
)

temp_attendance = attendance[(attendance["pct_attendance"] != "s")]

# Get school -> 2018 attendance mapping.
attendance_2018 = temp_attendance[temp_attendance.year == 2018]
attendance_2018 = dict(
    zip(attendance_2018.district, attendance_2018.pct_attendance.astype(float))
)

# Get school -> 2022 attendance mapping.
attendance_2022 = temp_attendance[temp_attendance.year == 2022]
attendance_2022 = dict(
    zip(attendance_2022.district, attendance_2022.pct_attendance.astype(float))
)


# Get difference in attendance.
def get_diff(district):
    return attendance_2022[district] - attendance_2018[district]


attendance_diff = pd.DataFrame(attendance.district.unique())
attendance_diff.columns = ["district"]
attendance_diff["attendance_chg_2018_2022"] = attendance_diff.district.apply(
    get_diff
)
attendance_diff["abs_change"] = attendance_diff.attendance_chg_2018_2022.abs()
attendance_diff.sort_values("abs_change", ascending=False, inplace=True)

with sns.axes_style("darkgrid"):
    for i in range(5):
        district = attendance_diff.iloc[i].district
        district_df = attendance[(attendance.district == district)][
            ["year", "pct_attendance"]
        ]
        plt.plot(
            district_df.year,
            district_df.pct_attendance.astype(float),
            label="District " + str(int(district)),
        )
plt.legend()
plt.ylabel("Percent Attendance (All Grades)")
plt.xticks([2018, 2019, 2020, 2021, 2022])
plt.title("Top Movers in Attendance (2018-2022)")
plt.gcf().text(
    0.1,
    0,
    "Schools with the largest change in percent attendance from 2018 to 2022.",
)
plt.savefig(
    "Output/attendance_time_series/top_district_movers.png",
    bbox_inches="tight",
)

fig, axs = plt.subplots(2, 2)
fig.suptitle("Top Movers in Attendance, All Grades (2018-2022)")
for i in range(4):
    district = attendance_diff.iloc[i].district
    district_df = attendance[(attendance.district == district)][
        ["year", "pct_attendance"]
    ]
    axs[i // 2, i % 2].plot(
        district_df.year,
        district_df.pct_attendance.astype(float),
        label="District " + str(int(district)),
    )
    axs[i // 2, i % 2].set_title("District " + str(int(district)))
    axs[i // 2, i % 2].set_ylabel("% Attendance")
    axs[i // 2, i % 2].set_xticks([2018, 2019, 2020, 2021, 2022])
    axs[i // 2, i % 2].set_xlabel("Year")
    axs[i // 2, i % 2].set_ylim(75, 95)
plt.tight_layout()
plt.savefig(
    "Output/attendance_time_series/top_district_movers_subplots.png",
)
