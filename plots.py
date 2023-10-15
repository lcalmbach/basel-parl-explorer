import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go

from helper import round_to_nearest


def line_chart(df, settings):
    title = settings["title"] if "title" in settings else ""
    if "x_dt" not in settings:
        settings["x_dt"] = "Q"
    if "y_dt" not in settings:
        settings["y_dt"] = "Q"
    if not ("opacity" in settings):
        settings["opacity"] = 1
    st.write(df.dtypes)
    y_axis = alt.Y(
        f"{settings['y']}:{settings['y_dt']}",
        scale=alt.Scale(domain=settings["y_domain"]),
        axis=alt.Axis(title=settings["y_title"]),
    )
    if "x_axis_values" in settings and settings["x_axis_values"]:
        x_axis = alt.X(
            f"{settings['x']}:{settings['x_dt']}",
            scale=alt.Scale(domain=settings["x_domain"]),
            tickCount="jahr",
        )
    else:
        x_axis = alt.X(
            f"{settings['x']}:{settings['x_dt']}",
            scale=alt.Scale(domain=settings["x_domain"]),
        )
    if "color" in settings and settings["color"] is not None:
        if "hide_legend" in settings:
            color = alt.Color(
                settings["color"], legend=None, scale=alt.Scale(domain=["grey"])
            )
        else:
            color = alt.Color(
                settings["color"], scale=alt.Scale(scheme="redblue", reverse=True)
            )
        chart = (
            alt.Chart(df)
            .mark_line(width=2, clip=True, opacity=settings["opacity"])
            .encode(
                x=x_axis,
                y=y_axis,
                color=color,
                tooltip=settings["tooltip"],
            )
        )
    else:
        chart = (
            alt.Chart(df)
            .mark_line(width=2, clip=True, opacity=settings["opacity"])
            .encode(
                x=x_axis,
                y=y_axis,
                tooltip=settings["tooltip"],
            )
        )

    if "compare_line" in settings:
        df2 = df[df["year"] == settings["compare_line"]]
        chart += (
            alt.Chart(df2)
            .mark_line(width=2, clip=True, color="red")
            .encode(
                x=alt.X(
                    f"{settings['x']}:{settings['x_dt']}",
                    scale=alt.Scale(domain=settings["x_domain"]),
                ),
                y=alt.Y(
                    f"{settings['y']}:{settings['y_dt']}",
                    scale=alt.Scale(domain=settings["y_domain"]),
                ),
                tooltip=settings["tooltip"],
            )
        )

    if "regression" in settings:
        line = chart.transform_regression(settings["x"], settings["y"]).mark_line()
        plot = (chart + line).properties(
            width=settings["width"], height=settings["height"], title=title
        )
    else:
        plot = chart.properties(
            width=settings["width"], height=settings["height"], title=title
        )
    st.altair_chart(plot)


def scatter_plot(df, settings):
    title = settings["title"] if "title" in settings else ""
    chart = (
        alt.Chart(df)
        .mark_circle(
            size=60,
        )
        .encode(
            x=alt.X(settings["x"], scale=alt.Scale(domain=settings["domain"])),
            y=alt.Y(settings["y"], scale=alt.Scale(domain=settings["domain"])),
            tooltip=settings["tooltip"],
            color=alt.Color(
                settings["color"], sort="ascending", scale=alt.Scale(scheme="bluered")
            ),
        )
        .interactive()
    )
    plot = chart.properties(
        width=settings["width"], height=settings["height"], title=title
    )
    st.altair_chart(plot)


def time_series_bar(df, settings):
    chart = (
        alt.Chart(df)
        .mark_bar(size=settings["size"], clip=True)
        .encode(
            x=alt.X(
                f"{settings['x']}:T",
                title=settings["x_title"],
                scale=alt.Scale(
                    domain=settings["x_domain"], axis=alt.Axis(format=("%b %Y"))
                ),
            ),
            y=alt.Y(f"{settings['y']}:Q", title=settings["y_title"]),
            tooltip=settings["tooltip"],
        )
    )
    plot = chart.properties(
        width=settings["width"], height=settings["height"], title=settings["title"]
    )
    st.altair_chart(plot)


def time_series_line(df, settings):
    if "x_domain" in settings:
        xax = alt.X(
            f"{settings['x']}:T",
            title=settings["x_title"],
            scale=alt.Scale(domain=settings["x_domain"]),
        )
    else:
        xax = alt.X(
            f"{settings['x']}:T",
            title=settings["x_title"],
        )

    if settings["y_domain"][0] != settings["y_domain"][1]:
        yax = alt.Y(
            f"{settings['y']}:Q",
            title=settings["y_title"],
            scale=alt.Scale(domain=settings["y_domain"]),
        )
    else:
        yax = alt.Y(f"{settings['y']}:Q", title=settings["y_title"])

    if "color" in settings:
        chart = (
            alt.Chart(df)
            .mark_line(clip=True)
            .encode(
                x=xax,
                y=yax,
                color=f"{settings['color']}:N",
                tooltip=settings["tooltip"],
            )
        )
    else:
        chart = (
            alt.Chart(df)
            .mark_line(clip=True)
            .encode(x=xax, y=yax, tooltip=settings["tooltip"])
        )

    if "h_line" in settings:
        chart += (
            alt.Chart(df)
            .mark_line(clip=True, color="red")
            .encode(x=xax, y=settings["h_line"], tooltip=settings["h_line"])
        )

    if "symbol_size" in settings:
        if not ("symbol_opacity" in settings):
            settings["symbol_opacity"] = 0.6
        if "color" in settings:
            chart += (
                alt.Chart(df)
                .mark_circle(
                    size=settings["symbol_size"],
                    clip=True,
                    opacity=settings["symbol_opacity"],
                )
                .encode(
                    x=xax,
                    y=yax,
                    color=f"{settings['color']}:N",
                    tooltip=settings["tooltip"],
                )
            )
        else:
            chart += (
                alt.Chart(df)
                .mark_circle(
                    size=settings["symbol_size"], opacity=settings["symbol_opacity"]
                )
                .encode(x=xax, y=yax, tooltip=settings["tooltip"])
            )
    plot = chart.properties(
        width=settings["width"], height=settings["height"], title=settings["title"]
    )
    st.altair_chart(plot)


def time_series_chart(df, settings):
    # line = alt.Chart(df_line).mark_line(color= 'red').encode(
    #    x= 'x',
    #    y= 'y'
    #    )

    title = settings["title"] if "title" in settings else ""
    if "x_title" not in settings:
        settings["x_title"] = ""
    if "symbol_size" not in settings:
        settings["symbol_size"] = 0
    if "rolling_avg_window" not in settings:
        settings["rolling_avg_window"] = 0
    plot = (
        alt.Chart(df)
        .mark_line(point=alt.OverlayMarkDef(color="blue", size=settings["symbol_size"]))
        .encode(
            x=alt.X(
                f"{settings['x']}:T", title=settings["x_title"]
            ),  # , scale=alt.Scale(domain=settings['x_domain']), ),
            y=alt.Y(
                f"{settings['y']}:Q",
                scale=alt.Scale(domain=settings["y_domain"]),
                title=settings["y_title"],
            ),
            tooltip=settings["tooltip"],
        )
    )
    if "show_regression" in settings:
        if len(df) > 2 and settings["show_regression"]:
            line = plot.transform_regression(settings["x"], settings["y"]).mark_line(
                color="orange"
            )
            plot += line
    if "show_average" in settings:
        if settings["show_average"]:
            avg = df[settings["y"]].mean()
            df_avg = pd.DataFrame(
                {
                    "x": [df[settings["x"]].min(), df[settings["x"]].max()],
                    "y": [avg, avg],
                }
            )
            line = (
                alt.Chart(df_avg)
                .mark_line(color="red")
                .encode(
                    x="x",
                    y="y",
                )
            )
            plot += line
    if settings["rolling_avg_window"] > 0:
        df["ma"] = (
            df[settings["y"]].rolling(window=settings["rolling_avg_window"]).mean()
        )
        # Create the chart
        line = (
            alt.Chart(df)
            .mark_line(color="green")
            .encode(x=f"{settings['x']}:T", y=f"ma:Q", strokeWidth=alt.value(3))
        )
        plot += line

    plot = plot.properties(
        width=settings["width"], height=settings["height"], title=title
    )
    st.altair_chart(plot)


def heatmap(df, settings):
    title = settings["title"] if "title" in settings else ""
    if not ("show_numbers" in settings):
        settings["show_numbers"] = True
    if not ("color_scheme" in settings):
        settings["color_scheme"] = "viridis"

    plot = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            # x=alt.X(settings["x"], sort=list(cn.MONTHS_REV_DICT.keys())),
            x=alt.X(settings["x"]),
            y=alt.Y(
                settings["y"],
                sort=alt.EncodingSortField(field="year", order="descending"),
            ),
            color=alt.Color(
                f"{settings['color']}:Q",
                scale=alt.Scale(range=["lightblue", "darkred"]),
            ),
            tooltip=settings["tooltip"],
        )
    )

    if settings["show_numbers"]:
        plot += plot.mark_text().encode(
            text=settings["color"], color=alt.value("black")
        )

    plot = plot.properties(width=settings["width"], title=title)
    st.altair_chart(plot)


def bar_chart(df: pd.DataFrame, settings: dict):
    if "title" not in settings:
        settings["title"] = ""
    if "tooltip" not in settings:
        settings["tooltip"] = [settings["x"], settings["y"]]
    if "bar_width" not in settings:
        settings["bar_width"] = 10
    if df[settings["x"]].dtype == "datetime64[ns]":
        x_axis = alt.X(
            f"{settings['x']}:T",
            axis=alt.Axis(title=settings["x_title"], format=settings["format_x"]),
        )
    else:
        x_axis = alt.X(f"{settings['x']}:{settings['x_dt']}")
    if "x_domain" in settings:
        x_axis.axis.scale = alt.Scale(domain=settings["x_domain"])
    y_axis = alt.Y(f"{settings['y']}:{settings['y_dt']}", title=settings["y_title"])
    if "y_domain" in settings:
        y_axis.axis.scale = alt.Scale(domain=settings["y_domain"])

    if "color_scheme" in settings:
        color_scheme = settings["color_scheme"]
        # not solved yet, see: https://stackoverflow.com/questions/66347857/sort-a-normalized-stacked-bar-chart-with-altair
        color = alt.Color(
            f"{settings['color']}:N",
            scale=alt.Scale(
                domain=list(color_scheme.keys()), range=list(color_scheme.values())
            ),
            sort=settings["x_sort"],
        )
    else:
        color = settings["color"]

    plot = (
        alt.Chart(df)
        .mark_bar(size=settings["bar_width"])
        .encode(x=x_axis, y=y_axis, color=color, tooltip=settings["tooltip"])
    )
    if "h_line" in settings:
        plot += (
            alt.Chart(df)
            .mark_line(color="red")
            .encode(
                x=x_axis,
                y=settings["h_line"],
            )
        )

    plot = plot.properties(
        title=settings["title"], width=settings["width"], height=settings["height"]
    )

    return st.altair_chart(plot)


def box_plot(df: pd.DataFrame, settings: dict):
    if "title" not in settings:
        settings["title"] = ""
    x_axis = alt.X(f"{settings['x']}:N", title=settings["x_title"])
    y_axis = alt.Y(settings["y"], title=settings["y_title"])
    plot = alt.Chart(df).mark_boxplot().encode(x=x_axis, y=y_axis)
    if "h_line" in settings:
        plot += (
            alt.Chart(df)
            .mark_line(color="red")
            .encode(
                x=f"{settings['x']}:N",
                y=settings["h_line"],
            )
        )

    plot = plot.properties(
        title=settings["title"], width=settings["width"], height=settings["height"]
    )

    return st.altair_chart(plot)


def histogram(df: pd.DataFrame, settings: dict):
    def get_x_domain():
        x_domain = [df[settings["x"]].min(), df[settings["x"]].max()]
        if x_domain[0] % 2 != 0:
            x_domain[0] -= 1
        if x_domain[1] % 2 != 0:
            x_domain[1] += 1
        return x_domain

    if "maxbins" not in settings:
        rounded_num = round_to_nearest(len(df), 10)
        settings["maxbins"] = rounded_num
    if "x_domain" not in settings:
        settings["x_domain"] = get_x_domain()
    if "title" not in settings:
        settings["title"] = ""

    plot = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{ settings['x'] }:Q",
                bin=alt.BinParams(maxbins=settings["maxbins"]),
                scale=alt.Scale(domain=settings["x_domain"]),
                title=settings["x_title"],
            ),
            y=alt.Y("count()", axis=alt.Axis(title=settings["y_title"])),
        )
    ).properties(
        title=settings["title"], width=settings["width"], height=settings["height"]
    )

    return st.altair_chart(plot)


def barchart(df, settings):
    title = settings["title"] if "title" in settings else ""
    """
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(f'year:N', axis=alt.Axis(title='', labelAngle=90)),
        y=alt.Y(f'stromverbrauch_kwh:Q', title=settings['y_title'], axis=alt.Axis(grid=False)),
        column = alt.Column('month:N',title=""),
        color='year:N',
        tooltip=settings['tooltip']
        ).configure_view(
            stroke=None,
        )
    """

    sort_field = settings["x"].replace(":Q", "")
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(
                settings["x"],
                title=settings["x_title"],
            ),
            y=alt.Y(
                settings["y"],
                title=settings["y_title"],
                sort=alt.EncodingSortField(field=sort_field, order="descending"),
            ),
            color=alt.condition(
                alt.datum.selected,
                alt.value("red"),  # highlight a bar with red.
                alt.value("lightgrey"),  # And grey for the rest of the bars
            ),
            tooltip=settings["tooltip"],
        )
    )
    if "h_line" in settings:
        chart += (
            alt.Chart(df)
            .mark_line()
            .encode(
                x=alt.X(
                    settings["x"],
                    title=settings["x_title"],
                    sort=alt.EncodingSortField(field=sort_field, order="descending"),
                ),
                y=alt.Y(settings["h_line"], title="Mittelwert"),
                color=alt.value("red"),
                tooltip=settings["h_line"],
            )
        )
    plot = chart.properties(
        width=settings["width"], height=settings["height"], title=title
    )
    st.altair_chart(plot)


def radar_chart(df: pd.DataFrame, settings: dict):
    """Shows a radar plot. Data is first normalized to a 0-100 scale.

    Args:
        df (pd.DataFrame)): Expected input: df with columns: ['name', *[name of variables]]
        settings (_type_): [title, ]
    """

    variables = list(df.columns).copy()
    variables.remove("name")
    fig = go.Figure()

    for index, row in df.iterrows():
        trace = []
        name = row["name"]
        for col in variables:
            trace.append(row[col])
        fig.add_trace(
            go.Scatterpolar(r=trace, theta=variables, fill="toself", name=name)
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=settings["range"])),
        showlegend=True,
    )
    st.plotly_chart(fig)
