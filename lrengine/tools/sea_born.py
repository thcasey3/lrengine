"""
class for plotting with seaborn
"""
import seaborn as sns


class sea:
    """
    sea class

    Attributes:
        df (DataFrame): "frame" from start object
        kind (str): type of plot
        options (dict): dictionary of plot options
    """

    def __init__(self, df, kind, options):

        if kind == "relplot":
            self.plt_relplot(df, options)
        else:
            raise ValueError("only relplot is currently supported")

    def plt_relplot(
        self,
        df,
        options={},
    ):
        opts = {
            "theme": "darkgrid",
            "hue": None,
            "size": None,
            "style": None,
            "row": None,
            "col": None,
            "col_wrap": None,
            "row_order": None,
            "col_order": None,
            "palette": None,
            "hue_order": None,
            "hue_norm": None,
            "sizes": None,
            "size_order": None,
            "size_norm": None,
            "markers": None,
            "dashes": None,
            "style_order": None,
            "legend": "auto",
            "kind": "scatter",
            "height": 5,
            "aspect": 1,
            "facet_kws": None,
            "units": None,
        }
        opts = self.update_opts(opts, options)
        sns.set_theme(style=opts["theme"])

        sns.relplot(
            data=df,
            x=opts["x"],
            y=opts["y"],
            hue=opts["hue"],
            s=opts["s"],
            size=opts["size"],
            style=opts["style"],
            row=opts["row"],
            col=opts["col"],
            col_wrap=opts["col_wrap"],
            row_order=opts["row_order"],
            col_order=opts["col_order"],
            palette=opts["palette"],
            hue_order=opts["hue_order"],
            hue_norm=opts["hue_norm"],
            sizes=opts["sizes"],
            size_order=opts["size_order"],
            size_norm=opts["size_norm"],
            markers=opts["markers"],
            dashes=opts["dashes"],
            style_order=opts["style_order"],
            legend="auto",
            kind="scatter",
            height=5,
            aspect=1,
            facet_kws=opts["facet_kws"],
            units=opts["units"],
        )

    @staticmethod
    def update_opts(opts, options):

        for ky in options.keys():
            opts[ky] = options[ky]

        return opts
