"""
class for plotting with seaborn
"""
import seaborn as sns


class sea:
    """
    sea class

    Attributes:
        df (DataFrame): "frame" from start object
        type (str): type of plot
        options (dict): dictionary of plot options
    """

    def __init__(self, df, type, options):

        if type == "lineplot":
            self.plt_lineplot(df, options)
        elif type == "histplot":
            self.plt_histplot(df, options)
        elif type == "catplot":
            self.plt_catplot(df, options)
        elif type == "lmplot":
            self.plt_lmplot(df, options)
        elif type == "replot":
            self.plt_replot(df, options)
        elif type == "scatterplot":
            self.plt_boxplot(df, options)
        elif type == "swarmplot":
            self.plt_swarmplot(df, options)
        elif type == "boxplot":
            self.plt_scatterplot(df, options)

    def plt_lineplot(
        self,
        df,
        options={},
    ):
        opts = {"style": "darkgrid", "plot_style": "event"}
        opts = update_opts(opts, options)

        sns.set_theme(style=opts["style"])
        sns.lineplot(
            data=df, x=opts["x"], y=opts["y"], hue=opts["hue"], style=opts["plot_style"]
        )

    def plt_lmplot(
        self,
        df,
        options={},
    ):
        opts = {
            "col": options["hue"],
            "style": "ticks",
            "col_wrap": 2,
            "ci": None,
            "palette": "muted",
            "height": 2,
            "scatter_kws": {"s": 50, "alpha": 1},
        }
        opts = update_opts(opts, options)

        sns.set_theme(style=opts["style"])
        sns.lmplot(
            data=df,
            x=opts["x"],
            y=opts["y"],
            hue=opts["hue"],
            col=opts["col"],
            col_wrap=opts["col_wrap"],
            ci=opts["ci"],
            palette=opts["palette"],
            height=opts["height"],
            scatter_kws=opts["scatter_kws"],
        )

    def plt_histplot(
        self,
        df,
        options={},
    ):
        opts = {
            "multiple": "stack",
            "style": "ticks",
            "edgecolor": ".3",
            "palette": "Spectral",
            "linewidth": 0.5,
            "log_scale": True,
        }
        opts = update_opts(opts, options)

        sns.set_theme(style=opts["style"])
        sns.histplot(
            data=df,
            x=opts["x"],
            hue=opts["hue"],
            multiple=opts["multiple"],
            palette=opts["palette"],
            edgecolor=opts["edgecolor"],
            linewidth=opts["linewidth"],
            log_scale=opts["log_scale"],
        )

    def plt_catplot(
        self,
        df,
        options={},
    ):
        opts = {
            "kind": "bar",
            "style": "whitegrid",
            "ci": "sd",
            "palette": "Spectral",
            "alpha": 0.6,
            "height": 6,
        }
        opts = update_opts(opts, options)

        sns.set_theme(style=opts["style"])
        sns.catplot(
            data=df,
            kind=opts["kind"],
            x=opts["x"],
            y=opts["y"],
            hue=opts["hue"],
            palette=opts["palette"],
            ci=opts["ci"],
            alpha=opts["alpha"],
            height=opts["height"],
        )

    def plt_replot(
        self,
        df,
        options={},
    ):
        opts = {
            "kind": "scatter",
            "style": "ticks",
            "font_scale": 1.25,
            "palette": "Spectral",
            "marker": "o",
            "s": 100,
        }
        opts = update_opts(opts, options)

        sns.set_theme(style=opts["style"], font_scale=opts["font_scale"])
        sns.relplot(
            data=df,
            kind=opts["kind"],
            x=opts["x"],
            y=opts["y"],
            hue=opts["hue"],
            palette=opts["palette"],
            marker=opts["marker"],
            s=opts["s"],
        )

    def plt_boxplot(
        self,
        df,
        options={},
    ):
        opts = {
            "style": "ticks",
            "palette": "Spectral",
        }
        opts = update_opts(opts, options)

        sns.set_theme(style=opts["style"])
        sns.boxplot(
            data=df, x=opts["x"], y=opts["y"], hue=opts["hue"], palette=opts["palette"]
        )

    def plt_scatterplot(
        self,
        df,
        options={},
    ):
        opts = {"color": ".15", "s": 5, "style": "whitegrid"}
        opts = update_opts(opts, options)

        sns.set_theme(style=opts["style"])
        sns.scatterplot(
            data=df,
            x=opts["x"],
            y=opts["y"],
            hue=opts["hue"],
            s=opts["s"],
            color=opts["color"],
        )

    def plt_swarmplot(
        self,
        df,
        options={},
    ):
        opts = {"style": "whitegrid", "palette": "muted"}
        opts = update_opts(opts, options)

        sns.set_theme(style=opts["style"], palette=opts["palette"])
        sns.swarmplot(data=df, x=opts["x"], y=opts["y"], hue=opts["hue"])

    @staticmethod
    def update_opts(opts, options):

        for ky in options.keys():
            opts[ky] = options[ky]

        return opts
