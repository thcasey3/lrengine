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

        if type == "replot":
            self.plt_replot(df, options)

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

        for ky in options.keys():
            opts[ky] = options[ky]

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
