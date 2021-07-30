def proc_epr(path, args):
    data = dnp.dnpImport.load(path)
    if "frequency" in data.attrs.keys():
        return [
            np.argmax(data.values[0], axis=0) / len(data.values[0]),
            data.attrs["frequency"],
        ]
