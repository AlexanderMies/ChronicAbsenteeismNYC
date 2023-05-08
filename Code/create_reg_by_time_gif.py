import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import imageio
import cv2

OUTDIR = "Output/correlation_narrative/"

# Load in attendance, test, and demographic data.
df = pd.read_csv("Data/Processed/absentee_tests_demographics_all_grades.csv")


def quick_plot2(
    x,
    y,
    xlab=None,
    ylab=None,
    filename=None,
    data=df,
    hue=None,
    reg=False,
    ylim=None,
    year_regs=[],
    **kwargs,
):
    xlab = x if xlab is None else xlab
    ylab = y if ylab is None else ylab
    title = f"{ylab} vs {xlab}"
    # plt.title(f"{ylab} vs {xlab}")

    # Set one standard figure size.
    plt.figure(figsize=(8, 6))

    with sns.axes_style("darkgrid"):
        if reg:
            sns.lmplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                legend=True,
                legend_out=False,
                line_kws=kwargs,
                scatter_kws={"s": 11},
                height=6,
                aspect=1.33,
            ).set(title=title)
            # Turn off the legend.
            # plt.gca().legend([], [], frameon=False)
        else:
            sns.scatterplot(data=data, x=x, y=y, hue=hue, palette="Set1").set(
                title=title
            )

            for year in year_regs:
                # Get the first three values from the Seaborn Set1 color palette.
                colors = {2018: "red", 2019: "blue", 2022: "green"}

                # Get data for the year.
                year_data = data[data["year"] == year]

                # Fit a regression line.
                year_model = smf.ols(
                    "pct_level_3and4_math ~ pct_attendance", data=year_data
                ).fit()

                # Get predictions.
                year_x_vals = np.linspace(
                    year_data["pct_attendance"].min(),
                    year_data["pct_attendance"].max(),
                    year_data.shape[0],
                )
                year_y_vals = year_model.predict(
                    pd.DataFrame({"pct_attendance": year_x_vals})
                )

                # Add regression line.
                plt.plot(year_x_vals, year_y_vals, color=colors[year])

        # Set one standard figure size.
        # plt.gcf().set_size_inches(8, 6)

        # Remove the legend title.
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.gca().legend(handles=handles[0:], labels=labels[0:])

        if ylim is not None:
            plt.ylim(ylim)

        # Set layout to tight.
        plt.tight_layout()

        plt.xlabel(xlab)
        plt.ylabel(ylab)

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename, bbox_inches="tight")
            print("Saved to", filename)
        plt.close()


quick_plot2(
    "pct_attendance",
    "pct_level_3and4_math",
    xlab="Pct Attendance",
    ylab="Pct Students Top Half Math",
    hue="year",
    reg=False,
    year_regs=[2018],
    ylim=(0, 80),
    filename=OUTDIR + f"attendance_vs_math_year_reg2018.png",
)

quick_plot2(
    "pct_attendance",
    "pct_level_3and4_math",
    xlab="Pct Attendance",
    ylab="Pct Students Top Half Math",
    hue="year",
    reg=False,
    year_regs=[2018, 2019],
    ylim=(0, 80),
    filename=OUTDIR + f"attendance_vs_math_year_reg2018_2019.png",
)

quick_plot2(
    "pct_attendance",
    "pct_level_3and4_math",
    xlab="Pct Attendance",
    ylab="Pct Students Top Half Math",
    hue="year",
    reg=False,
    year_regs=[2018, 2019, 2022],
    ylim=(0, 80),
    filename=OUTDIR + f"attendance_vs_math_year_reg2018_2019_2022.png",
)

# Create a gif of the above plots.
# The first image should be the plot with no regression lines (attendance_vs_math_year_scatter.png).
# The second image should be the plot with the 2018 regression line (attendance_vs_math_year_reg2018.png).
# The third image should be the plot with the 2018 and 2019 regression lines (attendance_vs_math_year_reg2018_2019.png).
# The fourth image should be the plot with the 2018, 2019, and 2022 regression lines (attendance_vs_math_year_reg2018_2019_2022.png).
# The fifth image should be the plot with the 2018, 2019, and 2022 regression lines (attendance_vs_math_year_reg.png).

plot_files = [
    # OUTDIR + f"attendance_vs_math_year_scatter.png",
    OUTDIR + f"attendance_vs_math_year_reg2018.png",
    OUTDIR + f"attendance_vs_math_year_reg2018_2019.png",
    OUTDIR + f"attendance_vs_math_year_reg2018_2019_2022.png",
    OUTDIR + f"attendance_vs_math_year_reg.png",
]

# Create a list of images for the gif.
images = []

for filename in plot_files:
    # Read in the image.
    image = imageio.imread(filename)

    # Resize the image to (600,800,4).
    image = cv2.resize(image, (800, 600))

    # Append the image to the list.
    images.append(image)


# Save the gif.
imageio.mimsave(
    OUTDIR + "attendance_vs_math_by_year.gif", images, duration=750
)
