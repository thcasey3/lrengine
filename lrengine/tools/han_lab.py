"""
a built in function as an example of a user defined function
"""
import sys
import os
import copy

sys.path.append("/Users/thomascasey/DNPLab")
import dnplab as dnp
import numpy as np


def optCenter(ws, width, starting_center, phase):
    optcenter_workspace = copy.deepcopy(ws)
    intgrl_array = []
    indx = range(starting_center - 50, starting_center + 50)
    optcenter_workspace["proc"].values *= np.exp(-1j * phase)
    for k in indx:
        iterativeopt_workspace = copy.deepcopy(optcenter_workspace)
        dnp.dnpTools.integrate(
            iterativeopt_workspace,
            integrate_center=k,
            integrate_width=width,
        )
        if len(iterativeopt_workspace["proc"].values) > 1:
            intgrl_array.append(sum(abs(iterativeopt_workspace["proc"].real.values)))
        else:
            intgrl_array.append(abs(iterativeopt_workspace["proc"].real.values[-1]))

    cent = np.argmax(intgrl_array)

    return indx[cent]


def optPhase(ws, width, starting_center, starting_phase):
    temp_data = ws["proc"][
        "f2", (starting_center - width, starting_center + width)
    ].values

    phases = np.linspace(
        starting_phase - np.pi / 2, starting_phase + np.pi / 2, 100
    ).reshape(1, -1)
    rotated_data = (temp_data.reshape(-1, 1)) * np.exp(-1j * phases)
    bestindex = np.argmax(
        (np.real(rotated_data) ** 2).sum(axis=0)
        / (np.imag(rotated_data) ** 2).sum(axis=0)
    )
    starting_phase = phases[0, bestindex]

    if ws["proc"].ndim == 2:

        phases = np.linspace(
            starting_phase - np.pi / 4,
            starting_phase + np.pi / 4,
            100,
        )
        imag_sum = []
        for indx, k in enumerate(phases):
            ws_rot = copy.deepcopy(ws)
            ws_rot["proc"].values *= np.exp(-1j * k)
            dnp.dnpTools.integrate(
                ws_rot,
                integrate_center=starting_center,
                integrate_width=width * 2,
            )
            imag_sum.append(np.sum(abs(ws_rot["proc"].imag.values * -1j)))

        starting_phase = phases[np.argmin(imag_sum)]

    base_data1 = ws["proc"][
        "f2",
        (
            (starting_center - width * 4),
            (starting_center - width / 2),
        ),
    ].values
    base_data2 = ws["proc"][
        "f2",
        (
            (starting_center + width / 2),
            (starting_center + width * 4),
        ),
    ].values
    base_data = np.concatenate((base_data2, base_data1))

    phases = np.linspace(
        starting_phase - np.pi / 4, starting_phase + np.pi / 4, 100
    ).reshape(1, -1)
    rotated_data = (base_data.reshape(-1, 1)) * np.exp(-1j * phases)
    bestindex = np.argmin(abs(np.real(rotated_data)).sum(axis=0))

    return phases[0, bestindex]


def optWidth(ws, starting_width, center, phase):
    ydata = abs(
        np.real(
            ws["proc"][
                "f2",
                (
                    center - starting_width / 2,
                    center + starting_width / 2,
                ),
            ].values
            * np.exp(-1j * phase)
        )
    )
    xdata = np.ravel(
        ws["proc"][
            "f2",
            (
                center - starting_width / 2,
                center + starting_width / 2,
            ),
        ].coords["f2"]
    )
    qual_factor = 1 / 3
    if ws["proc"].ndim == 1:
        one_third = np.where(ydata > max(ydata) * qual_factor)
        one_third = np.ravel(one_third)

        center = optCenter(
            ws, (xdata[one_third[-1]] - xdata[one_third[0]]), center, phase
        )

        return center, (xdata[one_third[-1]] - xdata[one_third[0]])
    else:
        min_x = []
        max_x = []
        for k in range(0, ydata.shape[1]):
            one_third = np.where(ydata[:, k] > max(ydata[:, k]) * qual_factor)
            one_third = np.ravel(one_third)
            min_x.append(xdata[one_third[0]])
            max_x.append(xdata[one_third[-1]])

        center = optCenter(ws, max(max_x) - min(min_x), center, phase)

        return center, max(max_x) - min(min_x)


def calc_odnp(path, in_dict):

    hyd = in_dict["hyd"]
    print("Working on: " + path)

    folder_structure_p0 = 5
    folder_structure_enh = range(6, 27)
    folder_structure_T1 = range(28, 33)
    folder_structure_T10 = 304

    E_power_List = dnp.dnpIO.cnsi.get_powers(
        path,
        "power",
        folder_structure_enh,
    )
    Epowers = np.add(E_power_List, 21.9992)
    Epowers = np.divide(Epowers, 10)
    Epowers = np.power(10, Epowers)
    Epowers = np.multiply(1e-3, Epowers)

    T1_power_List = dnp.dnpIO.cnsi.get_powers(
        path,
        "t1_powers",
        folder_structure_T1,
    )
    T1powers = np.add(T1_power_List, 21.9992)
    T1powers = np.divide(T1powers, 10)
    T1powers = np.power(10, T1powers)
    T1powers = np.multiply(1e-3, T1powers)

    folder_structure_all = []
    folder_structure_all.append(folder_structure_p0)
    for k in folder_structure_enh:
        folder_structure_all.append(k)
    for k in folder_structure_T1:
        folder_structure_all.append(k)
    folder_structure_all.append(folder_structure_T10)

    Ep = []
    T1 = []
    for _, folder_num in enumerate(folder_structure_all):

        folder_path = os.path.join(path, str(folder_num))

        data = dnp.dnpImport.load(folder_path)
        ws = dnp.create_workspace("proc", data)

        dnp.dnpNMR.remove_offset(ws)
        dnp.dnpNMR.window(
            ws,
            linewidth=10,
        )
        dnp.dnpNMR.fourier_transform(ws, zero_fill_factor=2)

        if ws["proc"].ndim == 2:
            dnp.dnpNMR.align(ws)
            max_index = np.argmax(abs(ws["proc"].values), axis=0)[-1]
        elif ws["proc"].ndim == 1:
            max_index = np.argmax(abs(ws["proc"].values), axis=0)

        starting_width = 10
        starting_center = round(ws["proc"].coords["f2"][max_index])
        starting_phase = np.arctan(
            np.sum(ws["proc"].imag.values) / np.sum(ws["proc"].real.values)
        )

        starting_phase = optPhase(ws, starting_width, starting_center, starting_phase)

        center = optCenter(
            ws,
            starting_width,
            starting_center,
            starting_phase,
        )

        phase = optPhase(
            ws,
            starting_width,
            center,
            starting_phase,
        )

        width = starting_width
        # center, width = optWidth(ws, starting_width, center, phase)

        ws["proc"].values *= np.exp(-1j * phase)

        dnp.dnpTools.integrate(
            ws,
            integrate_center=center,
            integrate_width=width,
        )

        if len(ws["proc"].values) > 1:
            dnp.dnpFit.exponential_fit(ws, type="T1")
            if folder_num == 304:
                hyd["T10"] = ws["fit"].attrs["T1"]
            else:
                T1.append(ws["fit"].attrs["T1"])
        else:
            if folder_num == 5:
                p0 = ws["proc"].real.values[0]
            else:
                Ep.append(ws["proc"].real.values[0] / p0)

    hyd.update(
        {
            "E": np.array(Ep),
            "E_power": np.array(Epowers),
            "T1": np.array(T1),
            "T1_power": np.array(T1powers),
        }
    )

    hydra = dnp.create_workspace()
    hydra.add("hydration_inputs", hyd)

    try:
        hydration_results = dnp.dnpHydration.hydration(hydra)
    except:
        hydration_results = {"ksigma": 0, "tcorr": 0}

    print("Found ksigma = " + str(hydration_results["ksigma"]))
    print("Found tcorr = " + str(hydration_results["tcorr"]))

    return hydration_results["ksigma"], hydration_results["tcorr"]
