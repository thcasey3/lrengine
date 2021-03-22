# import seaborn as sns


class sea:
    def __init__(self, df, kind, options):

        if kind == "replot":
            self.plt_replot(df, options)

    def plt_replot(self, df, options):

        print(df.shape)
        print("REPLOT ", options["test"])
