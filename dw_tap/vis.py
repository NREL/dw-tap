import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_monthly_avg(atmospheric_df, ws_column="ws", datetime_column="datetime",
                     title="Windspeed monthly averages",
                     show_avg_across_years=True,
                     label_avg_across_years=True,
                     save_to_file=True,
                     show_overall_avg=True,
                     show=True):

    if ws_column not in atmospheric_df.columns:
        raise ValueError("Can't find %s column in dataframe. Skipping plotting" % ws_column)
    if datetime_column not in atmospheric_df.columns:
        raise ValueError("Can't find %s column in dataframe. Skipping plotting" % datetime_column)

    df = atmospheric_df[[datetime_column, ws_column]].copy()

    year_month = pd.Series(pd.PeriodIndex(df[datetime_column], freq="M"))
    df["month"] = year_month.apply(lambda x: x.month)
    df["year"] = year_month.apply(lambda x: x.year)
    #display(df)

    fig, ax = plt.subplots(figsize=(10, 3))
    xvals = list(range(1, 13)) # for showing monthly data

    for year, grp in df.groupby("year"):
        monthly_avg = grp[[ws_column, "month"]].groupby("month").agg(np.mean)
        ax.plot(monthly_avg, label=str(year), linestyle="--")

    if show_avg_across_years:
        monthly_avg_across_years = df.groupby("month")[ws_column].agg(np.mean)
        ax.plot(monthly_avg_across_years, label="Avg across years", marker="o")
        if label_avg_across_years:
            ylim0 = ax.get_ylim()[0]
            ylim1 = ax.get_ylim()[1]
            yoffset = ylim1 / 35  # express offest as a fraction of height
            yvals = pd.Series(monthly_avg_across_years.tolist())
            a = pd.concat({'x': pd.Series(xvals),
                           'y': yvals,
                           'val': yvals}, axis=1)
            for i, point in a.iterrows():
                t = ax.text(point['x'], point['y'] + yoffset, "%.2f" % point['val'], fontsize=7)
                t.set_bbox(dict(facecolor='lightgray', alpha=0.75, edgecolor='red'))
            ax.set_ylim([ylim0, ylim1*1.25])

    ax.set_xticks(xvals)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    ax.set_ylabel("Monthly avg windspeed, m/s")
    ax.set_title(title)

    plt.figtext(0.1, -0.05, 
                "Code used to produce this figure is developed under NREL's TAP project "
                "(https://www.nrel.gov/wind/tools-assessing-performance.html)", 
                ha="left", fontsize=6)
    
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));

    if show_overall_avg:
        ax.set_xlim([0, 16])
        overall_avg = df[ws_column].mean()
        ax.axhline(y=overall_avg,linestyle="dotted", color="orange", linewidth=2.0)
        t = ax.text(13.0, overall_avg + yoffset, "Overall avg=%.2f" % overall_avg, fontsize=8)
        t.set_bbox(dict(facecolor='orange', alpha=0.3, edgecolor='black'))
        
    if save_to_file == True:
        plt.savefig('%s.png' % title, dpi=300)
    elif type(save_to_file) == str:
        plt.savefig(save_to_file, dpi=300)

    if show:
        plt.show()