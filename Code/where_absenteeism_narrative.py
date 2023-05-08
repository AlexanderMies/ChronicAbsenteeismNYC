import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

OUTDIR = "Output/correlation_narrative/"

# Load in attendance, test, and demographic data.
df = pd.read_csv("Data/Processed/absentee_tests_demographics_all_grades.csv")
df.columns
df["pct_nonwhite"] = 1 - df["pct_white"]
df["black_hispanic"] = df["pct_black"] + df["pct_hispanic"]
df["is_black_hispanic"] = np.where(
    df["black_hispanic"] > df["black_hispanic"].median(),
    "$> 77\%$ Black or Hispanic",
    "$\leq 77\%$ Black or Hispanic",
)
df["is_poverty"] = np.where(
    df["pct_poverty"] > df["pct_poverty"].median(),
    "$> 76\%$ Poverty",
    "$\leq 76\%$ Poverty",
)
df["is_ela"] = np.where(
    df["pct_english_language_learners"]
    > df["pct_english_language_learners"].median(),
    "$> 11\%$ English Language Learners",
    "$\leq 11\%$ English Language Learners",
)
df["high_chronic_absenteeism"] = np.where(
    df["pct_chronically_absent"] > df["pct_chronically_absent"].median(),
    "$> 32\%$ Chronic Absenteeism",
    "$\leq 32\%$ Chronic Absenteeism",
)


df["black"] = np.where(
    df["pct_black"] > df["pct_black"].median(), "black", "not black"
)
df["hispanic"] = np.where(
    df["pct_hispanic"] > df["pct_hispanic"].median(),
    "hispanic",
    "not hispanic",
)


def quick_plot(
    x,
    y,
    xlab=None,
    ylab=None,
    filename=None,
    data=df,
    hue=None,
    reg=False,
    ylim=None,
    add_lines_model=False,
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
            sns.scatterplot(data=data, x=x, y=y, hue=hue).set(title=title)

            # add manual lines if specified.
            if add_lines_model:

                def predict(model, x_col, dummy_val):
                    if dummy_val == 1:
                        dummy_val = data[data[hue].str.contains(">")][
                            hue
                        ].iloc[0]
                    else:
                        dummy_val = data[~data[hue].str.contains(">")][
                            hue
                        ].iloc[0]
                    return model.predict(
                        pd.DataFrame({x: x_col, hue: dummy_val})
                    )

                x_vals = np.linspace(
                    data[x].min(), data[x].max(), data.shape[0]
                )

                # Add regression line for dummy var = 0.
                plt.plot(
                    x_vals,
                    predict(add_lines_model, x_vals, 0),
                    color=sns.color_palette()[0],
                )

                # Add regression line for dummy var = 1.
                plt.plot(
                    x_vals,
                    predict(add_lines_model, x_vals, 1),
                    color=sns.color_palette()[1],
                )

        # Set one standard figure size.
        # plt.gcf().set_size_inches(8, 6)

        # Remove the legend title.
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.gca().legend(handles=handles[1:], labels=labels[1:])

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


# Checking chronically absent vs level 3 and 4 math.
# quick_plot(
#     "pct_chronically_absent",
#     "pct_level_3and4_math",
#     xlab="Pct Attendance",
#     ylab="Pct Students Top Half Math",
#     # filename=OUTDIR + "attendance_vs_math_all.png",
#     ylim=(0, 80),
# )

############
# Section 1: Score vs Attendance
############

# First plot: pct attendance vs pct level 3 and 4 math.
quick_plot(
    "pct_attendance",
    "pct_level_3and4_math",
    xlab="Pct Attendance",
    ylab="Pct Students Top Half Math",
    filename=OUTDIR + "attendance_vs_math_all.png",
    ylim=(0, 80),
)
# Same plot with reg line
smf.ols(
    "pct_level_3and4_math ~ pct_attendance", data=df
).fit().summary()  # 4.2 pct points of students in top half of math for every 1 pct point increase in attendance.
quick_plot(
    "pct_attendance",
    "pct_level_3and4_math",
    xlab="Pct Attendance",
    ylab="Pct Students Top Half Math",
    filename=OUTDIR + "attendance_vs_math_all_regplot.png",
    reg=True,
    color="red",
    ylim=(0, 80),
)

############
# Section 2: Score vs Attendance, with dummy vars
############
# Plot with only dummy var difference.
for split_var in [
    "is_black_hispanic",
    "is_poverty",
    "high_chronic_absenteeism",
]:
    model = smf.ols(
        "pct_level_3and4_math ~ pct_attendance + {}".format(split_var), data=df
    ).fit()

    # Print the model equation nicely.
    print("Model equation for", split_var, "is:")
    print(
        "score ~ {} + {} * attendance + {} * {}".format(
            *model.params, split_var
        )
    )
    print(model.summary())

    quick_plot(
        "pct_attendance",
        "pct_level_3and4_math",
        xlab="Pct Attendance",
        ylab="Pct Students Top Half Math",
        filename=OUTDIR
        + "attendance_vs_math_{}_custom_reg.png".format(split_var),
        hue=split_var,
        reg=False,
        ylim=(0, 80),
        add_lines_model=model,
    )


variables = [
    "is_black_hispanic",
    "is_poverty",
    "high_chronic_absenteeism",
    "is_ela",
]
suffixes = ["blackHispanic", "poverty", "chronicAbsenteeism", "ela"]
for variable, suffix in zip(variables, suffixes):
    for reg in [True, False]:
        quick_plot(
            "pct_attendance",
            "pct_level_3and4_math",
            xlab="Pct Attendance",
            ylab="Pct Students Top Half Math",
            hue=variable,
            reg=reg,
            ylim=(0, 80),
            filename=OUTDIR
            + f"attendance_vs_math_{variable}_regplot={reg}.png",
        )

############
# Section 3: Score vs Attendance, with dummy vars and interaction terms
############


def map_binary_text(s):
    return int(">" in s)


# The lmplots take care of plotting the regressions, so we just need to get the model equations.
for split_var in [
    "is_black_hispanic",
    "is_poverty",
    "high_chronic_absenteeism",
]:
    temp_df = df.copy()
    temp_df[split_var] = temp_df[split_var].apply(map_binary_text)
    model = smf.ols(
        "pct_level_3and4_math ~ pct_attendance + {} + pct_attendance * {}".format(
            split_var, split_var
        ),
        data=temp_df,
    ).fit()

    # Print the model equation nicely.
    print("Model equation for", split_var, "is:")
    print(
        "score ~ {} + {} * attendance + {} * {} + {} * attendance * {}".format(
            *model.params, split_var, split_var
        )
    )
    print(model.summary())

############
# Section 4: Does year play a role?
############

# Not going to use the quick_plot() function b/c three groups makes it a bit more complicated.

# First do scatterplot.
with sns.axes_style("darkgrid"):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x="pct_attendance",
        y="pct_level_3and4_math",
        hue="year",
        data=df,
        palette="Set1",
    )
    plt.xlabel("Pct Attendance")
    plt.ylabel("Pct Students Top Half Math")
    plt.title("Pct Attendance vs Pct Students Top Half Math")

    # Remove the legend title.
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles=handles[0:], labels=labels[0:])

    plt.ylim([0, 80])

    plt.tight_layout()
    plt.savefig(OUTDIR + "attendance_vs_math_year_scatter.png")
    plt.close()
    # plt.show()

# Then do regression.
with sns.axes_style("darkgrid"):
    # plt.figure(figsize=(8, 6))
    sns.lmplot(
        x="pct_attendance",
        y="pct_level_3and4_math",
        hue="year",
        data=df,
        palette="Set1",
        height=6,
        aspect=1.33,
        legend=False,
    )
    plt.xlabel("Pct Attendance")
    plt.ylabel("Pct Students Top Half Math")
    plt.title("Pct Attendance vs Pct Students Top Half Math")

    # Remove the legend title.
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.gca().legend(handles=handles[0:], labels=labels[0:])

    plt.ylim([0, 80])

    plt.tight_layout()
    plt.savefig(OUTDIR + "attendance_vs_math_year_reg.png")
    plt.close()
    # plt.show()


######
# Double Check Regression Equations
######

# score ~ attendance
model = smf.ols("pct_level_3and4_math ~ pct_attendance", data=df).fit()
print("Model equation for score ~ attendance is:")
print("score ~ {} + {} * attendance".format(*model.params))

# score ~ attendance + is_black_hispanic
df["black_hispanic_binary"] = df.is_black_hispanic.apply(
    map_binary_text
).where(df.is_black_hispanic.notnull(), np.nan)
model = smf.ols(
    "pct_level_3and4_math ~ pct_attendance + black_hispanic_binary", data=df
).fit()
print("Model equation for score ~ attendance + is_black_hispanic is:")
print(model.summary())

# score ~ attendance + is_poverty
df["poverty_binary"] = df.is_poverty.apply(map_binary_text).where(
    df.is_poverty.notnull(), np.nan
)
model = smf.ols(
    "pct_level_3and4_math ~ pct_attendance + poverty_binary", data=df
).fit()
print("Model equation for score ~ attendance + is_poverty is:")
print(model.summary())

# score ~ attendance + high_chronic_absenteeism
df["chronic_absenteeism_binary"] = df.high_chronic_absenteeism.apply(
    map_binary_text
).where(df.high_chronic_absenteeism.notnull(), np.nan)
model = smf.ols(
    "pct_level_3and4_math ~ pct_attendance + chronic_absenteeism_binary",
    data=df,
).fit()
print("Model equation for score ~ attendance + high_chronic_absenteeism is:")
print(model.summary())

# score ~ attendance + is_black_hispanic + attendance * is_black_hispanic
model = smf.ols(
    "pct_level_3and4_math ~ pct_attendance + black_hispanic_binary + pct_attendance * black_hispanic_binary",
    data=df,
).fit()
print(
    "Model equation for score ~ attendance + is_black_hispanic + attendance * is_black_hispanic is:"
)
print(model.summary())

# score ~ attendance + is_poverty + attendance * is_poverty
model = smf.ols(
    "pct_level_3and4_math ~ pct_attendance + poverty_binary + pct_attendance * poverty_binary",
    data=df,
).fit()
print(
    "Model equation for score ~ attendance + is_poverty + attendance * is_poverty is:"
)
print(model.summary())

# score ~ attendance + high_chronic_absenteeism + attendance * high_chronic_absenteeism
model = smf.ols(
    "pct_level_3and4_math ~ pct_attendance + chronic_absenteeism_binary + pct_attendance * chronic_absenteeism_binary",
    data=df,
).fit()
print(
    "Model equation for score ~ attendance + high_chronic_absenteeism + attendance * high_chronic_absenteeism is:"
)
print(model.summary())

# score ~ attendance in 2018
model = smf.ols(
    "pct_level_3and4_math ~ pct_attendance", data=df.loc[df.year == 2018]
).fit()
print("Model equation for score ~ attendance in 2018 is:")
print(model.summary())

# score ~ attendance in 2019
model = smf.ols(
    "pct_level_3and4_math ~ pct_attendance", data=df.loc[df.year == 2019]
).fit()
print("Model equation for score ~ attendance in 2019 is:")
print(model.summary())

# score ~ attendance in 2022
model = smf.ols(
    "pct_level_3and4_math ~ pct_attendance", data=df.loc[df.year == 2022]
).fit()
print("Model equation for score ~ attendance in 2022 is:")
print(model.summary())
