import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_monthly_avg(atmospheric_df, ws_column="ws", datetime_column="datetime", 
                     title="Windspeed monthly averages",
                     show_avg_across_years=True,
                     label_avg_across_years=True,
                     save_to_file=True,
                     show=True):

    if ws_column not in atmospheric_df.columns:
        print("Can't find required %s column in the given dataframe. Skipping plotting" % ws_column)
        return
    if datetime_column not in atmospheric_df.columns:
        print("Can't find required %s column in the given dataframe. Skipping plotting" % datetime_column)
        return
    
    df = atmospheric_df[[datetime_column, ws_column]].copy()
    
    year_month = pd.Series(pd.PeriodIndex(df['datetime'], freq="M"))
    df["month"] = year_month.apply(lambda x: x.month)
    df["year"] = year_month.apply(lambda x: x.year)

    fig, ax = plt.subplots(figsize=(11, 3))
    xvals = list(range(1, 13)) # for showing monthly data
    
    for year, grp in df.groupby("year"):
        monthly_avg = grp[["ws", "month"]].groupby("month").agg(np.mean)
        ax.plot(monthly_avg, label=str(year), linestyle="--")

    if show_avg_across_years:
        monthly_avg_across_years = df.groupby("month")[ws_column].agg(np.mean)
        ax.plot(monthly_avg_across_years, label="Avg across years", marker="o")
        if label_avg_across_years:
            ylim0 = ax.get_ylim()[0]
            ylim1 = ax.get_ylim()[1]
            yoffset = ylim1 / 25  # express offest as a fraction of height
            yvals = pd.Series(monthly_avg_across_years.tolist())
            a = pd.concat({'x': pd.Series(xvals), 
                           'y': yvals, 
                           'val': yvals}, axis=1)
            for i, point in a.iterrows():
                t = ax.text(point['x'], point['y'] + yoffset, "%.2f" % point['val'])
                t.set_bbox(dict(facecolor='lightgray', alpha=0.75, edgecolor='red'))
            ax.set_ylim([ylim0, ylim1*1.25])
        
    ax.set_xticks(xvals)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])

    ax.set_ylabel("Monthly avg windspeed, m/s")
    ax.set_title(title)

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5));
    
    if save_to_file == True:
        plt.savefig('%s.png' % title, dpi=300)
    elif type(save_to_file) == str:
        plt.savefig(save_to_file, dpi=300) 

    if show:
    	plt.show()
