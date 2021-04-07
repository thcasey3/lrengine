"""
sea_born module, for making a relplot with seaborn
"""
import seaborn as sns


class sea:
    """
    class for interacting with seaborn

    Args:
        df (DataFrame): "frame" from start object
        kind (str): type of plot (currently only relplot is allowed)
        seaborn_args (dict): dict keys are the seaborn.relplot arguments and their allowed values are according to seaborn documentation for relplot
    Returns:
        seaborn relplot
    """

    def __init__(self, df=None, kind="relplot", seaborn_args={}):

        if kind == "relplot":
            self.plt_relplot(df=df, kind=kind, seaborn_args=seaborn_args)
        else:
            raise ValueError("only relplot is currently supported")

    def plt_relplot(
        self,
        df,
        kind,
        seaborn_args,
    ):
        sea_args = {
            "theme": "darkgrid",
            "hue": None,
            "s": 25,
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
        sea_args = self.update_sea_args(sea_args, seaborn_args)
        sns.set_theme(style=sea_args["theme"])

        sns.relplot(
            data=df,
            x=sea_args["x"],
            y=sea_args["y"],
            hue=sea_args["hue"],
            s=sea_args["s"],
            size=sea_args["size"],
            style=sea_args["style"],
            row=sea_args["row"],
            col=sea_args["col"],
            col_wrap=sea_args["col_wrap"],
            row_order=sea_args["row_order"],
            col_order=sea_args["col_order"],
            palette=sea_args["palette"],
            hue_order=sea_args["hue_order"],
            hue_norm=sea_args["hue_norm"],
            sizes=sea_args["sizes"],
            size_order=sea_args["size_order"],
            size_norm=sea_args["size_norm"],
            markers=sea_args["markers"],
            dashes=sea_args["dashes"],
            style_order=sea_args["style_order"],
            legend="auto",
            kind="scatter",
            height=5,
            aspect=1,
            facet_kws=sea_args["facet_kws"],
            units=sea_args["units"],
        )

    @staticmethod
    def update_sea_args(sea_args, seaborn_args):

        for ky in seaborn_args.keys():
            sea_args[ky] = seaborn_args[ky]

        return sea_args
