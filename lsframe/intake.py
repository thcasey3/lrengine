"""
intake module, classes for finding dates and patterns
"""

from . import engine
from datetime import date
from dateutil import parser
import re
import numpy as np


class date_injectors:
    """
    class for looking for dates in file or folder names

    Args:
        lsdata (start): start object

    Returns:
        updated start object
    """

    def __init__(self, lsdata):

        if lsdata.date_format:
            if lsdata.date_format == "any" or isinstance(lsdata.date_format, list):
                self._smart_search_dates(lsdata)
            elif isinstance(lsdata.date_format, str):
                self._look_for_dates(lsdata)

    def _smart_search_dates(self, lsdata):

        date_list = list(np.zeros(len(lsdata.frame)).astype(int))
        date_format_list = list(np.zeros(len(lsdata.frame)).astype(int))
        date_delta_list = list(np.zeros(len(lsdata.frame)).astype(int))

        if lsdata.date_format == "any":
            possible_formats = [
                "YYYYMMDD",
                "YYYYDDMM",
                "MMDDYYYY",
                "DDMMYYYY",
                "YYMMDD",
                "YYDDMM",
                "MMDDYY",
                "DDMMYY",
                "YYYYMDD",
                "YYYYDDM",
                "MDDYYYY",
                "DDMYYYY",
                "YYMDD",
                "YYDDM",
                "MDDYY",
                "DDMYY",
                "YYYYMMD",
                "YYYYDMM",
                "MMDYYYY",
                "DMMYYYY",
                "YYMMD",
                "YYDMM",
                "MMDYY",
                "DMMYY",
                "YYYYMD",
                "YYYYDM",
                "MDYYYY",
                "DMYYYY",
                "YYMD",
                "YYDM",
                "MDYY",
                "DMYY",
                "YYYY-MM-DD",
                "YYYY-DD-MM",
                "MM-DD-YYYY",
                "DD-MM-YYYY",
                "YY-MM-DD",
                "YY-DD-MM",
                "MM-DD-YY",
                "DD-MM-YY",
                "YYYY-M-DD",
                "YYYY-DD-M",
                "M-DD-YYYY",
                "DD-M-YYYY",
                "YY-M-DD",
                "YY-DD-M",
                "M-DD-YY",
                "DD-M-YY",
                "YYYY-MM-D",
                "YYYY-D-MM",
                "MM-D-YYYY",
                "D-MM-YYYY",
                "YY-MM-D",
                "YY-D-MM",
                "MM-D-YY",
                "D-MM-YY",
                "YYYY-M-D",
                "YYYY-D-M",
                "M-D-YYYY",
                "D-M-YYYY",
                "YY-M-D",
                "YY-D-M",
                "M-D-YY",
                "D-M-YY",
            ]
        elif isinstance(lsdata.date_format, list):
            possible_formats = lsdata.date_format

        for indx, direct in enumerate(lsdata.frame.name):
            possible_date = []
            possible_delta = []
            possible_patt = []
            for _, patt in enumerate(possible_formats):
                date_try = self._look_for_date_string(direct, patt)
                if date_try:
                    found_date = self.parse_dates(date_try)
                    if found_date is not None:
                        found_delta = self.diff_dates(found_date)
                        if (
                            found_delta >= 0
                            and 1900 < found_date.year < date.today().year
                        ):
                            possible_date.append(found_date)
                            possible_delta.append(found_delta)
                            possible_patt.append(patt)
                    else:
                        continue

            if possible_date:
                date_list[indx] = possible_date
                if date_list[indx]:
                    date_delta_list[indx] = possible_delta
                    date_format_list[indx] = possible_patt

        if not date_delta_list:
            print("No dates were found in names")
        else:
            lsdata.frame["date"] = date_list
            lsdata.frame["date_format"] = date_format_list
            lsdata.frame["date_delta"] = date_delta_list

        return lsdata

    def _look_for_dates(self, lsdata):

        date_list = list(np.zeros(len(lsdata.frame)).astype(int))
        date_delta_list = list(np.zeros(len(lsdata.frame)).astype(int))

        for indx, direct in enumerate(lsdata.frame.name):
            possible_date = self._look_for_date_string(direct, lsdata.date_format)
            if possible_date:
                date_list[indx] = self.parse_dates(possible_date)
                if date_list[indx] is not None:
                    if date_list[indx]:
                        date_delta_list[indx] = self.diff_dates(date_list[indx])
                else:
                    continue

        if sum(date_delta_list) != 0:
            lsdata.frame["date"] = date_list
            lsdata.frame["date_delta"] = date_delta_list
        else:
            print("No dates of format " + lsdata.date_format + " found in names")

        return lsdata

    def _look_for_date_string(self, direct, date_format):

        for x in ["/", "_", ":", ";"]:
            date_format = date_format.replace(x, "-")

        if date_format in [
            "YYYY-MM-DD",
            "YYYY-DD-MM",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{4}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{2}))", direct
            )
        elif date_format in [
            "DD-MM-YYYY",
            "MM-DD-YYYY",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{2}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{4}))", direct
            )
        elif date_format in [
            "YYYY-MM-D",
            "YYYY-DD-M",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{4}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{1}))", direct
            )
        elif date_format in [
            "DD-M-YYYY",
            "MM-D-YYYY",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{2}[^a-zA-Z0-9][0-9]{1}[^a-zA-Z0-9][0-9]{4}))", direct
            )
        elif date_format in [
            "M-DD-YYYY",
            "D-MM-YYYY",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{1}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{4}))", direct
            )
        if date_format in [
            "YYYY-M-DD",
            "YYYY-D-MM",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{4}[^a-zA-Z0-9][0-9]{1}[^a-zA-Z0-9][0-9]{2}))", direct
            )
        if date_format in [
            "YYYY-M-D",
            "YYYY-D-M",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{4}[^a-zA-Z0-9][0-9]{1}[^a-zA-Z0-9][0-9]{1}))", direct
            )
        elif date_format in [
            "D-M-YYYY",
            "M-D-YYYY",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{1}[^a-zA-Z0-9][0-9]{1}[^a-zA-Z0-9][0-9]{4}))", direct
            )
        elif date_format in [
            "YY-MM-DD",
            "YY-DD-MM",
            "DD-MM-YY",
            "MM-DD-YY",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{2}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{2}))", direct
            )
        elif date_format in [
            "YY-M-DD",
            "YY-D-MM",
            "DD-M-YY",
            "MM-D-YY",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{2}[^a-zA-Z0-9][0-9]{1}[^a-zA-Z0-9][0-9]{2}))", direct
            )
        elif date_format in ["YY-MM-D", "YY-DD-M"]:
            matches = re.finditer(
                r"(?=([0-9]{2}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{1}))", direct
            )
        elif date_format in [
            "YY-M-D",
            "YY-D-M",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{2}[^a-zA-Z0-9][0-9]{1}[^a-zA-Z0-9][0-9]{1}))", direct
            )
        elif date_format in [
            "D-M-YY",
            "M-D-YY",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{1}[^a-zA-Z0-9][0-9]{1}[^a-zA-Z0-9][0-9]{2}))", direct
            )
        elif date_format in [
            "D-MM-YY",
            "M-DD-YY",
        ]:
            matches = re.finditer(
                r"(?=([0-9]{1}[^a-zA-Z0-9][0-9]{2}[^a-zA-Z0-9][0-9]{2}))", direct
            )
        elif date_format in ["YYYYMMDD", "YYYYDDMM", "DDMMYYYY", "MMDDYYYY"]:
            matches = re.finditer(r"(?=(\d{8}))", direct)
        elif date_format in [
            "YYYYMMD",
            "YYYYDMM",
            "DMMYYYY",
            "MMDYYYY",
            "YYYYMDD",
            "YYYYDDM",
            "DDMYYYY",
            "MDDYYYY",
        ]:
            matches = re.finditer(r"(?=(\d{7}))", direct)
        elif date_format in [
            "YYYYMD",
            "YYYYDM",
            "DMYYYY",
            "MDYYYY",
            "YYMMDD",
            "YYDDMM",
            "MMDDYY",
            "DDMMYY",
        ]:
            matches = re.finditer(r"(?=(\d{6}))", direct)
        elif date_format in [
            "YYMMD",
            "YYDMM",
            "YYMDD",
            "YYDDM",
            "DDMYY",
            "MDDYY",
            "MMDYY",
            "MDDYY",
            "DMMYY",
        ]:
            matches = re.finditer(r"(?=(\d{5}))", direct)
        elif date_format in ["YYMD", "YYDM", "DMYY", "MDYY"]:
            matches = re.finditer(r"(?=(\d{4}))", direct)

        date_string = [match.group(1) for match in matches]

        if date_string and isinstance(date_string, list):
            new_date_string = []
            for ds in date_string:
                ds.replace("/", "-").replace("_", "-").replace(":", "-")
                test_date_string = self._map_date_string(ds, date_format)
                test = self.parse_dates(test_date_string)
                if test is not None:
                    new_date_string.append(test_date_string)
                else:
                    continue
            if new_date_string and len(new_date_string) == 1:
                date_string = new_date_string[0]
            elif new_date_string and len(new_date_string) > 1:
                date_string = [x for x in new_date_string if x != []]

        return date_string

    def _map_date_string(self, date_string, date_format):

        if date_format == "YYYY-MM-DD":
            date_string = (
                date_string[0:4] + "-" + date_string[5:7] + "-" + date_string[8:]
            )
        if date_format == "YYYY-DD-MM":
            date_string = (
                date_string[0:4] + "-" + date_string[8:] + "-" + date_string[5:7]
            )

        if date_format == "DD-MM-YYYY":
            date_string = (
                date_string[6:] + "-" + date_string[3:5] + "-" + date_string[0:2]
            )

        if date_format == "MM-DD-YYYY":
            date_string = (
                date_string[6:] + "-" + date_string[0:2] + "-" + date_string[3:5]
            )

        if date_format == "YYYYMMDD":
            date_string = date_string[0:4] + date_string[4:6] + date_string[6:]

        if date_format == "YYYYDDMM":
            date_string = date_string[0:4] + date_string[6:] + date_string[4:6]

        if date_format == "DDMMYYYY":
            date_string = date_string[4:] + date_string[2:4] + date_string[0:2]

        if date_format == "MMDDYYYY":
            date_string = date_string[4:] + date_string[0:4]

        if date_format == "YY-MM-DD":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[0:2]
                    + "-"
                    + date_string[3:5]
                    + "-"
                    + date_string[6:]
                )
            else:
                date_string = (
                    "20"
                    + date_string[0:2]
                    + "-"
                    + date_string[3:5]
                    + "-"
                    + date_string[6:]
                )

        if date_format == "YY-DD-MM":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[0:2]
                    + "-"
                    + date_string[6:]
                    + "-"
                    + date_string[3:5]
                )
            else:
                date_string = (
                    "20"
                    + date_string[0:2]
                    + "-"
                    + date_string[6:]
                    + "-"
                    + date_string[3:5]
                )

        if date_format == "DD-MM-YY":
            if int("20" + date_string[6:]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[6:]
                    + "-"
                    + date_string[3:5]
                    + "-"
                    + date_string[0:2]
                )
            else:
                date_string = (
                    "20"
                    + date_string[6:]
                    + "-"
                    + date_string[3:5]
                    + "-"
                    + date_string[0:2]
                )

        if date_format == "MM-DD-YY":
            if int("20" + date_string[6:]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[6:]
                    + "-"
                    + date_string[0:2]
                    + "-"
                    + date_string[3:5]
                )
            else:
                date_string = (
                    "20"
                    + date_string[6:]
                    + "-"
                    + date_string[0:2]
                    + "-"
                    + date_string[3:5]
                )

        if date_format == "YYMMDD":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19" + date_string[0:2] + date_string[2:4] + date_string[4:]
                )
            else:
                date_string = (
                    "20" + date_string[0:2] + date_string[2:4] + date_string[4:]
                )

        if date_format == "YYDDMM":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19" + date_string[0:2] + date_string[4:] + date_string[2:4]
                )
            else:
                date_string = (
                    "20" + date_string[0:2] + date_string[4:] + date_string[2:4]
                )

        if date_format == "DDMMYY":
            if int("20" + date_string[4:]) > date.today().year:
                date_string = (
                    "19" + date_string[4:] + date_string[2:4] + date_string[0:2]
                )
            else:
                date_string = (
                    "20" + date_string[4:] + date_string[2:4] + date_string[0:2]
                )

        if date_format == "MMDDYY":
            if int("20" + date_string[4:]) > date.today().year:
                date_string = "19" + date_string[4:] + date_string[0:4]
            else:
                date_string = "20" + date_string[4:] + date_string[0:4]

        if date_format == "YYYY-M-DD":
            date_string = (
                date_string[0:4] + "-" + "0" + date_string[5:6] + "-" + date_string[7:]
            )
        if date_format == "YYYY-DD-M":
            date_string = (
                date_string[0:4] + "-" + "0" + date_string[8:] + "-" + date_string[5:7]
            )

        if date_format == "DD-M-YYYY":
            date_string = (
                date_string[5:] + "-" + "0" + date_string[3:4] + "-" + date_string[0:2]
            )

        if date_format == "M-DD-YYYY":
            date_string = (
                date_string[5:] + "-" + "0" + date_string[0:1] + "-" + date_string[2:4]
            )

        if date_format == "YYYYMDD":
            date_string = date_string[0:4] + "0" + date_string[4:5] + date_string[5:]

        if date_format == "YYYYDDM":
            date_string = date_string[0:4] + "0" + date_string[6:] + date_string[4:6]

        if date_format == "DDMYYYY":
            date_string = date_string[3:] + "0" + date_string[2:3] + date_string[0:2]

        if date_format == "MDDYYYY":
            date_string = date_string[3:] + "0" + date_string[0:3]

        if date_format == "YY-M-DD":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[0:2]
                    + "-"
                    + "0"
                    + date_string[3:4]
                    + "-"
                    + date_string[5:]
                )
            else:
                date_string = (
                    "20"
                    + date_string[0:2]
                    + "-"
                    + "0"
                    + date_string[3:4]
                    + "-"
                    + date_string[5:]
                )

        if date_format == "YY-DD-M":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[0:2]
                    + "-"
                    + "0"
                    + date_string[6:]
                    + "-"
                    + date_string[3:5]
                )
            else:
                date_string = (
                    "20"
                    + date_string[0:2]
                    + "-"
                    + "0"
                    + date_string[6:]
                    + "-"
                    + date_string[3:5]
                )

        if date_format == "DD-M-YY":
            if int("20" + date_string[5:]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[5:]
                    + "-"
                    + "0"
                    + date_string[3:4]
                    + "-"
                    + date_string[0:2]
                )
            else:
                date_string = (
                    "20"
                    + date_string[5:]
                    + "-"
                    + "0"
                    + date_string[3:4]
                    + "-"
                    + date_string[0:2]
                )

        if date_format == "M-DD-YY":
            if int("20" + date_string[5:]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[5:]
                    + "-"
                    + "0"
                    + date_string[0:1]
                    + "-"
                    + date_string[2:4]
                )
            else:
                date_string = (
                    "20"
                    + date_string[5:]
                    + "-"
                    + "0"
                    + date_string[0:1]
                    + "-"
                    + date_string[2:4]
                )

        if date_format == "YYMDD":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19" + date_string[0:2] + "0" + date_string[2:4] + date_string[4:]
                )
            else:
                date_string = (
                    "20" + date_string[0:2] + "0" + date_string[2:4] + date_string[4:]
                )

        if date_format == "YYDDM":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19" + date_string[0:2] + "0" + date_string[4:] + date_string[2:4]
                )
            else:
                date_string = (
                    "20" + date_string[0:2] + "0" + date_string[4:] + date_string[2:4]
                )

        if date_format == "DDMYY":
            if int("20" + date_string[3:]) > date.today().year:
                date_string = (
                    "19" + date_string[3:] + "0" + date_string[2:3] + date_string[0:2]
                )
            else:
                date_string = (
                    "20" + date_string[3:] + "0" + date_string[2:3] + date_string[0:2]
                )

        if date_format == "MDDYY":
            if int("20" + date_string[3:]) > date.today().year:
                date_string = "19" + date_string[3:] + "0" + date_string[0:3]
            else:
                date_string = "20" + date_string[3:] + "0" + date_string[0:3]

        if date_format == "YYYY-MM-D":
            date_string = (
                date_string[0:4] + "-" + date_string[5:7] + "-" + "0" + date_string[8:]
            )
        if date_format == "YYYY-D-MM":
            date_string = (
                date_string[0:4] + "-" + date_string[7:] + "-" + "0" + date_string[5:6]
            )

        if date_format == "D-MM-YYYY":
            date_string = (
                date_string[5:] + "-" + date_string[2:4] + "-" + "0" + date_string[0:1]
            )

        if date_format == "MM-D-YYYY":
            date_string = (
                date_string[5:] + "-" + date_string[0:2] + "-" + "0" + date_string[3:4]
            )

        if date_format == "YYYYMMD":
            date_string = date_string[0:4] + date_string[4:6] + "0" + date_string[6:]

        if date_format == "YYYYDMM":
            date_string = date_string[0:4] + date_string[5:] + "0" + date_string[4:5]

        if date_format == "DMMYYYY":
            date_string = date_string[3:] + date_string[1:3] + "0" + date_string[0:1]

        if date_format == "MMDYYYY":
            date_string = date_string[3:] + date_string[0:2] + "0" + date_string[2:3]

        if date_format == "YY-MM-D":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[0:2]
                    + "-"
                    + date_string[3:5]
                    + "-"
                    + "0"
                    + date_string[6:]
                )
            else:
                date_string = (
                    "20"
                    + date_string[0:2]
                    + "-"
                    + date_string[3:5]
                    + "-"
                    + "0"
                    + date_string[6:]
                )

        if date_format == "YY-D-MM":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[0:2]
                    + "-"
                    + date_string[5:]
                    + "-"
                    + "0"
                    + date_string[3:4]
                )
            else:
                date_string = (
                    "20"
                    + date_string[0:2]
                    + "-"
                    + date_string[5:]
                    + "-"
                    + "0"
                    + date_string[3:4]
                )

        if date_format == "D-MM-YY":
            if int("20" + date_string[5:]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[5:]
                    + "-"
                    + date_string[2:4]
                    + "-"
                    + "0"
                    + date_string[0:1]
                )
            else:
                date_string = (
                    "20"
                    + date_string[5:]
                    + "-"
                    + date_string[2:4]
                    + "-"
                    + "0"
                    + date_string[0:1]
                )

        if date_format == "MM-D-YY":
            if int("20" + date_string[5:]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[5:]
                    + "-"
                    + date_string[0:2]
                    + "-"
                    + "0"
                    + date_string[3:4]
                )
            else:
                date_string = (
                    "20"
                    + date_string[5:]
                    + "-"
                    + date_string[0:2]
                    + "-"
                    + "0"
                    + date_string[3:4]
                )

        if date_format == "YYMMD":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19" + date_string[0:2] + date_string[2:4] + "0" + date_string[4:]
                )
            else:
                date_string = (
                    "20" + date_string[0:2] + date_string[2:4] + "0" + date_string[4:]
                )

        if date_format == "YYDMM":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19" + date_string[0:2] + date_string[3:] + "0" + date_string[2:3]
                )
            else:
                date_string = (
                    "20" + date_string[0:2] + date_string[3:] + "0" + date_string[2:3]
                )

        if date_format == "DMMYY":
            if int("20" + date_string[3:]) > date.today().year:
                date_string = (
                    "19" + date_string[3:] + date_string[1:3] + "0" + date_string[0:1]
                )
            else:
                date_string = (
                    "20" + date_string[3:] + date_string[1:3] + "0" + date_string[0:1]
                )

        if date_format == "MMDYY":
            if int("20" + date_string[3:]) > date.today().year:
                date_string = (
                    "19" + date_string[3:] + date_string[0:2] + "0" + date_string[2:3]
                )
            else:
                date_string = (
                    "20" + date_string[3:] + date_string[0:2] + "0" + date_string[2:3]
                )

        if date_format == "YYYY-M-D":
            date_string = (
                date_string[0:4]
                + "-"
                + "0"
                + date_string[5:6]
                + "-"
                + "0"
                + date_string[7:]
            )
        if date_format == "YYYY-D-M":
            date_string = (
                date_string[0:4]
                + "-"
                + "0"
                + date_string[7:]
                + "-"
                + "0"
                + date_string[5:6]
            )
        if date_format == "D-M-YYYY":
            date_string = (
                date_string[4:]
                + "-"
                + "0"
                + date_string[2:3]
                + "-"
                + "0"
                + date_string[0:1]
            )

        if date_format == "M-D-YYYY":
            date_string = (
                date_string[4:]
                + "-"
                + "0"
                + date_string[0:1]
                + "-"
                + "0"
                + date_string[2:3]
            )

        if date_format == "YYYYMD":
            date_string = (
                date_string[0:4] + "0" + date_string[4:5] + "0" + date_string[5:]
            )

        if date_format == "YYYYDM":
            date_string = (
                date_string[0:4] + "0" + date_string[5:] + "0" + date_string[4:5]
            )

        if date_format == "DMYYYY":
            date_string = (
                date_string[2:] + "0" + date_string[1:2] + "0" + date_string[0:1]
            )

        if date_format == "MDYYYY":
            date_string = (
                date_string[2:] + "0" + date_string[0:1] + "0" + date_string[1:2]
            )

        if date_format == "YY-M-D":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[0:2]
                    + "-"
                    + "0"
                    + date_string[3:4]
                    + "-"
                    + "0"
                    + date_string[5:]
                )
            else:
                date_string = (
                    "20"
                    + date_string[0:2]
                    + "-"
                    + "0"
                    + date_string[3:4]
                    + "-"
                    + "0"
                    + date_string[5:]
                )

        if date_format == "YY-D-M":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[0:2]
                    + "-"
                    + "0"
                    + date_string[5:]
                    + "-"
                    + "0"
                    + date_string[3:4]
                )
            else:
                date_string = (
                    "20"
                    + date_string[0:2]
                    + "-"
                    + "0"
                    + date_string[5:]
                    + "-"
                    + "0"
                    + date_string[3:4]
                )

        if date_format == "D-M-YY":
            if int("20" + date_string[4:]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[4:]
                    + "-"
                    + "0"
                    + date_string[2:3]
                    + "-"
                    + "0"
                    + date_string[0:1]
                )
            else:
                date_string = (
                    "20"
                    + date_string[4:]
                    + "-"
                    + "0"
                    + date_string[2:3]
                    + "-"
                    + "0"
                    + date_string[0:1]
                )

        if date_format == "M-D-YY":
            if int("20" + date_string[4:]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[4:]
                    + "-"
                    + "0"
                    + date_string[0:1]
                    + "-"
                    + "0"
                    + date_string[2:3]
                )
            else:
                date_string = (
                    "20"
                    + date_string[4:]
                    + "-"
                    + "0"
                    + date_string[0:1]
                    + "-"
                    + "0"
                    + date_string[2:3]
                )

        if date_format == "YYMD":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[0:2]
                    + "0"
                    + date_string[2:3]
                    + "0"
                    + date_string[3:]
                )
            else:
                date_string = (
                    "20"
                    + date_string[0:2]
                    + "0"
                    + date_string[2:3]
                    + "0"
                    + date_string[3:]
                )

        if date_format == "YYDM":
            if int("20" + date_string[0:2]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[0:2]
                    + "0"
                    + date_string[3:]
                    + "0"
                    + date_string[2:3]
                )
            else:
                date_string = (
                    "20"
                    + date_string[0:2]
                    + "0"
                    + date_string[3:]
                    + "0"
                    + date_string[2:3]
                )

        if date_format == "DMYY":
            if int("20" + date_string[2:]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[2:]
                    + "0"
                    + date_string[1:2]
                    + "0"
                    + date_string[0:1]
                )
            else:
                date_string = (
                    "20"
                    + date_string[2:]
                    + "0"
                    + date_string[1:2]
                    + "0"
                    + date_string[0:1]
                )

        if date_format == "MDYY":
            if int("20" + date_string[2:]) > date.today().year:
                date_string = (
                    "19"
                    + date_string[2:]
                    + "0"
                    + date_string[0:1]
                    + "0"
                    + date_string[1:2]
                )
            else:
                date_string = (
                    "20"
                    + date_string[2:]
                    + "0"
                    + date_string[0:1]
                    + "0"
                    + date_string[1:2]
                )

        return date_string

    @staticmethod
    def parse_dates(possible_date):
        try:
            parsed = parser.isoparse(possible_date).date()
            return parsed
        except:
            return None

    @staticmethod
    def diff_dates(possible_date):
        delta = date.today() - date(
            possible_date.year, possible_date.month, possible_date.day
        )
        return int(delta.days)


class dates_filter:
    """
    class for reducing dates lists from the "all" option

    Args:
        lsdata (start): start object
        remove (str or list): format(s) of dates to remove
        keep (str or list): format(s) of dates to keep
        only_unique (bool): keep or remove duplicate dates
        strip_zeros (bool): keep or remove date=0 elements
        which (str): which method is using the functions

    Returns:
        updated start object
    """

    def __init__(
        self,
        lsdata,
        remove=None,
        keep=None,
        only_unique=True,
        strip_zeros=False,
        which=None,
    ):

        if which is not None:

            if remove is not None and keep is not None:
                raise ValueError("Please give either keep or remove, not both at once")

            self.strip_zeros = strip_zeros
            if strip_zeros:
                keep_indx = lsdata.frame[lsdata.frame["date"] != 0].index.tolist()
                lsdata.frame = lsdata.frame.loc[keep_indx, :]

            if isinstance(remove, str):
                remove = [remove]
            if isinstance(keep, str):
                keep = [keep]

            if which == "reduce":
                self._take_out_dates(
                    lsdata, remove=remove, keep=keep, only_unique=only_unique
                )
            elif which == "ondate":
                self._on_date(lsdata, remove=remove, keep=keep)
            elif which == "inrange":
                self._in_range(lsdata, remove=remove, keep=keep)

    @staticmethod
    def on_date_list(lsdata, conditions_list):

        if any([isinstance(x, list) for x in conditions_list]):
            raise ValueError(
                "for on_date() the elements of the list can not be lists, did you mean to use in_range()?"
            )

        for indx, kept in enumerate(conditions_list):
            if kept == "today":
                conditions_list[indx] = date.today().strftime("%Y-%m-%d")
            elif isinstance(kept, date):
                conditions_list[indx] = kept.strftime("%Y-%m-%d")
            elif isinstance(kept, str):
                try:
                    _ = parser.parse(kept)
                except:
                    raise ValueError("one of the dates is not a valid date, see docs.")

        keep_indx = []
        for indx, dates in enumerate(lsdata.frame["date"]):
            if isinstance(dates, list):
                dates = [x.strftime("%Y-%m-%d") for x in dates]
            elif isinstance(dates, date):
                dates = [dates.strftime("%Y-%m-%d")]

            if dates == 0 or any(map(dates.__contains__, conditions_list)):
                keep_indx.append(lsdata.frame.index[indx])

        return keep_indx

    def _on_date(self, lsdata, remove, keep):

        keep_indx = []
        if keep:
            keep_indx = self.on_date_list(lsdata, keep)

        if remove:
            rem_indx = self.on_date_list(lsdata, remove)
            for indx in lsdata.frame.index:
                if not self.strip_zeros:
                    if indx not in rem_indx or lsdata.frame.loc[indx, "date"] == 0:
                        keep_indx.append(indx)
                else:
                    if indx not in rem_indx:
                        keep_indx.append(indx)

        if keep_indx:
            keep_indx = list(set(keep_indx))
            keep_indx.sort()
            lsdata.frame = lsdata.frame.loc[keep_indx, :]
            return lsdata
        else:
            raise ValueError("the given options would return an empty frame")

    @staticmethod
    def in_range_list(lsdata, conditions_list):

        if not isinstance(conditions_list, list) or not all(
            [
                isinstance(x, str) or (isinstance(x, list) and len(x) == 2)
                for x in conditions_list
            ]
        ):
            raise ValueError(
                "for in_range() you must give only two element lists, did you mean to use on_date()?"
            )
        keep_indx = []
        if (
            isinstance(conditions_list, list)
            and len(conditions_list) == 2
            and not all([isinstance(x, list) for x in conditions_list])
        ):
            conditions_list = [
                x.replace("today", date.today().strftime("%Y-%m-%d"))
                for x in conditions_list
            ]
            try:
                _ = [parser.parse(x) for x in conditions_list]
            except:
                raise ValueError("one of the dates is not valid, see docs.")

            for indx, dates in enumerate(lsdata.frame["date"]):
                if isinstance(dates, list):
                    dates = [x.strftime("%Y-%m-%d") for x in dates]
                elif isinstance(dates, date):
                    dates = [dates.strftime("%Y-%m-%d")]

                if dates == 0 or any(
                    [conditions_list[0] < x < conditions_list[1] for x in dates]
                ):
                    keep_indx.append(lsdata.frame.index[indx])

        elif isinstance(conditions_list, list) and all(
            [isinstance(x, list) and len(x) == 2 for x in conditions_list]
        ):
            conditions_list = [
                [x.replace("today", date.today().strftime("%Y-%m-%d")) for x in y]
                for y in conditions_list
            ]
            try:
                _ = [[parser.parse(x) for x in y] for y in conditions_list]
            except:
                raise ValueError("one of the dates is not valid, see docs.")

            for indx, dates in enumerate(lsdata.frame["date"]):
                if isinstance(dates, list):
                    dates = [x.strftime("%Y-%m-%d") for x in dates]
                elif isinstance(dates, date):
                    dates = [dates.strftime("%Y-%m-%d")]

                for kept in conditions_list:
                    if dates == 0 or any([kept[0] < x < kept[1] for x in dates]):
                        keep_indx.append(lsdata.frame.index[indx])
        else:
            raise TypeError("the given list is not valid, see docs.")

        return keep_indx

    def _in_range(self, lsdata, remove, keep):

        keep_indx = []
        if keep:
            keep_indx = self.in_range_list(lsdata, keep)

        if remove:
            rem_indx = self.in_range_list(lsdata, remove)
            for indx in lsdata.frame.index:
                if not self.strip_zeros:
                    if indx not in rem_indx or lsdata.frame.loc[indx, "date"] == 0:
                        keep_indx.append(indx)
                else:
                    if indx not in rem_indx:
                        keep_indx.append(indx)

        if keep_indx:
            keep_indx = list(set(keep_indx))
            keep_indx.sort()
            lsdata.frame = lsdata.frame.loc[keep_indx, :]
            return lsdata
        else:
            raise ValueError("the given options would return an empty frame")

    def _take_out_dates(self, lsdata, remove, keep, only_unique):

        if "date_format" in lsdata.frame.columns:
            new_formats = list(np.zeros(len(lsdata.frame)).astype(int))
            new_dates = list(np.zeros(len(lsdata.frame)).astype(int))
            new_deltas = list(np.zeros(len(lsdata.frame)).astype(int))
            for indx, form_list in enumerate(lsdata.frame["date_format"]):
                if remove is None and keep is None:
                    new_formats[indx] = lsdata.frame.loc[
                        lsdata.frame.index[indx], "date_format"
                    ]
                    new_dates[indx] = lsdata.frame.loc[lsdata.frame.index[indx], "date"]
                    new_deltas[indx] = lsdata.frame.loc[
                        lsdata.frame.index[indx], "date_delta"
                    ]
                else:
                    if isinstance(form_list, list):
                        if remove is not None and keep is None:
                            new_formats[indx] = [
                                x
                                for x in lsdata.frame.loc[
                                    lsdata.frame.index[indx], "date_format"
                                ]
                                if x not in remove
                            ]
                            new_dates[indx] = [
                                lsdata.frame.loc[lsdata.frame.index[indx], "date"][x]
                                for x, form in enumerate(
                                    lsdata.frame.loc[
                                        lsdata.frame.index[indx], "date_format"
                                    ]
                                )
                                if form not in remove
                            ]
                            new_deltas[indx] = [
                                lsdata.frame.loc[
                                    lsdata.frame.index[indx], "date_delta"
                                ][x]
                                for x, form in enumerate(
                                    lsdata.frame.loc[
                                        lsdata.frame.index[indx], "date_format"
                                    ]
                                )
                                if form not in remove
                            ]
                        elif remove is None and keep is not None:
                            new_formats[indx] = [
                                x
                                for x in lsdata.frame.loc[
                                    lsdata.frame.index[indx], "date_format"
                                ]
                                if x in keep
                            ]
                            new_dates[indx] = [
                                lsdata.frame.loc[lsdata.frame.index[indx], "date"][x]
                                for x, form in enumerate(
                                    lsdata.frame.loc[
                                        lsdata.frame.index[indx], "date_format"
                                    ]
                                )
                                if form in keep
                            ]
                            new_deltas[indx] = [
                                lsdata.frame.loc[
                                    lsdata.frame.index[indx], "date_delta"
                                ][x]
                                for x, form in enumerate(
                                    lsdata.frame.loc[
                                        lsdata.frame.index[indx], "date_format"
                                    ]
                                )
                                if form in keep
                            ]

                    elif isinstance(form_list, str):
                        if remove is not None and keep is None:
                            if form_list not in remove:
                                new_formats[indx] = lsdata.frame.loc[
                                    lsdata.frame.index[indx], "date_format"
                                ]
                                new_dates[indx] = lsdata.frame.loc[
                                    lsdata.frame.index[indx], "date"
                                ]
                                new_deltas[indx] = lsdata.frame.loc[
                                    lsdata.frame.index[indx], "date_delta"
                                ]
                        elif remove is None and keep is not None:
                            if form_list in keep:
                                new_formats[indx] = lsdata.frame.loc[
                                    lsdata.frame.index[indx], "date_format"
                                ]
                                new_dates[indx] = lsdata.frame.loc[
                                    lsdata.frame.index[indx], "date"
                                ]
                                new_deltas[indx] = lsdata.frame.loc[
                                    lsdata.frame.index[indx], "date_delta"
                                ]

                if not new_dates[indx]:
                    new_dates[indx] = 0
                    new_formats[indx] = 0
                    new_deltas[indx] = 0
                elif isinstance(new_dates[indx], list) and len(new_dates[indx]) == 1:
                    new_dates[indx] = new_dates[indx][0]
                    new_formats[indx] = new_formats[indx][0]
                    new_deltas[indx] = new_deltas[indx][0]
                elif isinstance(new_dates[indx], list) and len(new_dates[indx]) > 1:
                    if only_unique:
                        nd = []
                        ndd = []
                        ndf = []
                        for ix, d in enumerate(new_dates[indx]):
                            if d not in nd:
                                nd.append(d)
                                ndd.append(new_deltas[indx][ix])
                                ndf.append(new_formats[indx][ix])

                        new_dates[indx] = nd
                        new_formats[indx] = ndf
                        new_deltas[indx] = ndd

            lsdata.frame["date_format"] = new_formats
            lsdata.frame["date"] = new_dates
            lsdata.frame["date_delta"] = new_deltas

        return lsdata


class pattern_injectors:
    """
    class for looking for patterns in file or folder names

    Args:
        lsdata (start): start object

    Returns:
        updated start object
    """

    def __init__(self, lsdata):

        if lsdata.patterns:
            if isinstance(lsdata.patterns, str):
                lsdata.patterns = [lsdata.patterns]
            self._look_for_patterns(lsdata)

    def _look_for_patterns(self, lsdata):

        if isinstance(lsdata.patterns, list):
            patterns_to_iterate = lsdata.patterns
        elif isinstance(lsdata.patterns, dict):
            patterns_to_iterate = lsdata.patterns.keys()

        for patt in patterns_to_iterate:
            lsdata.frame[patt] = list(np.zeros(len(lsdata.frame)).astype(bool))
            for indx, direct in enumerate(lsdata.frame["name"]):
                if isinstance(lsdata.patterns, list):
                    if patt in direct:
                        lsdata.frame.loc[indx, patt] = True

                elif isinstance(lsdata.patterns, dict):
                    value = lsdata.patterns[patt]
                    if value is bool:
                        if patt in direct:
                            lsdata.frame.loc[indx, patt] = True
                    else:
                        match = re.findall(value, direct)
                        if match:
                            if patt in match[0]:
                                found = match[0].replace(patt, "")
                            else:
                                found = match[0]
                            lsdata.frame.loc[indx, patt] = found

        return lsdata


class patterns_filter:
    """
    class for filtering names based on patterns

    Args:
        lsdata (start): start object
        remove (str or list): pattern(s) to use to decide which names to remove
        keep (str or list): pattern(s) to use to decide which names to keep

    Returns:
        updated start object
    """

    def __init__(self, lsdata, remove=None, keep=None):

        if isinstance(remove, str):
            remove = [remove]
        if isinstance(keep, str):
            keep = [keep]

        self._take_out_names(lsdata, remove=remove, keep=keep)

    def _take_out_names(self, lsdata, remove, keep):

        if remove is not None or keep is not None:
            keep_indx = []
            for indx, subdir in enumerate(lsdata.frame["name"]):
                if remove is not None and keep is None:
                    if not any(map(subdir.__contains__, remove)):
                        keep_indx.append(lsdata.frame.index[indx])
                elif remove is None and keep is not None:
                    if any(map(subdir.__contains__, keep)):
                        keep_indx.append(lsdata.frame.index[indx])
                elif remove is not None and keep is not None:
                    if not any(map(subdir.__contains__, remove)) or any(
                        map(subdir.__contains__, keep)
                    ):
                        keep_indx.append(lsdata.frame.index[indx])

            if len(keep_indx) == 0:
                raise TypeError(
                    "You removed all of your names! Try different remove or keep patterns"
                )
            else:
                lsdata.frame = lsdata.frame.loc[keep_indx, :]

        return lsdata
