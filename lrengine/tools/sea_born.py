# import seaborn as sns


class sea:
    def __init__(self, df, kind, options):

        if kind == "replot":
            self.plt_replot(df, options)

    def plt_replot(self):

        print("REPLOT")
