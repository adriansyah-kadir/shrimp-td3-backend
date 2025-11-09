from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from datetime import datetime
from water_env3 import WaterEnv
import pandas as pd
import requests


def timestamp(date: str, time: str):
    return int(datetime.strptime(f"{date} {time}", "%d/%m/%Y %H.%M.%S").timestamp())


class StateData:
    def __init__(self) -> None:
        self.env = WaterEnv()
        self.google_cred = service_account.Credentials.from_service_account_file(
            "/home/aldo/Work/shrimp-td3-backend/nodal-magnet-464414-h8-bc57f1e4d322.json",
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        self.google_req = Request()
        self.google_sheet = build("sheets", "v4", credentials=self.google_cred)

    def read_spreadsheet(self, spreadsheetId, range):
        self.google_cred.refresh(self.google_req)
        url = f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheetId}/values/{range}"
        headers = {"Authorization": f"Bearer {self.google_cred.token}"}
        resp = requests.get(url, headers=headers)
        data = resp.json()
        return data["values"]

    def read1(self) -> pd.DataFrame:
        "ph,suhu,do"
        spreadsheet_id = "1QNxnPYfcq2ARFocbFCbU5eBRoOndzV4AwA3vmjwSZxo"
        spreadsheet_range = "A:H"
        result = self.read_spreadsheet(spreadsheet_id, spreadsheet_range)
        columns = result[0]
        values = result[1:]
        df = pd.DataFrame(values, columns=columns)
        df["datetime"] = pd.to_datetime(
            df["Timestamp"] + " " + df["Jam"], format="%d/%m/%Y %H.%M.%S", errors="coerce",
        )
        df["timestamp"] = df["datetime"].astype(int) // 10**6  # convert ns → ms
        df["temp"] = df["Suhu DS18B20 (°C)"].astype(float)
        df["pH"] = df["pH"].astype(float)
        df["DO"] = df["DO (mg/L)"].astype(float)
        return df[["timestamp", "temp", "pH", "DO"]]  # pyright: ignore

    def read2(self) -> pd.DataFrame:
        spreadsheet_id = "1HTajyzTQEObhQX5Li0_MTlZ8dmrufjl9u1uB0VBieK8"
        spreadsheet_range = "A:I"
        result = self.read_spreadsheet(spreadsheet_id, spreadsheet_range)
        columns = result[0]
        values = result[1:]
        df = pd.DataFrame(values, columns=columns)
        df["datetime"] = pd.to_datetime(
            df["Tanggal dan waktu"] + " " + df["Jam"], format="%d/%m/%Y %H.%M.%S", errors="coerce"
        )
        df["timestamp"] = df["datetime"].astype(int) // 10**6  # convert ns → ms
        df["sal"] = df["konduktivitas"].astype(float) * 0.0008
        df["nh3"] = df['nh3_voltage'].astype(float)
        df["turb"] = df["turbidity_voltage"].astype(float)
        # df['turb'] = (
        #     -1120.4 * (df['turbidity_voltage'].astype(float) ** 2)
        #     + 5742.3 * df['turbidity_voltage'].astype(float)
        #     - 4352.9
        # )
        return df[["timestamp", "sal", "nh3", "turb"]]  # pyright: ignore

    def read(self) -> pd.DataFrame:
        df1 = self.read1()
        df2 = self.read2()
        if len(df1) < len(df2):
            df1["sal"] = df2["sal"]
            df1["nh3"] = df2["nh3"]
            df1["turb"] = df2["turb"]
        else:
            df2["temp"] = df1["temp"]
            df2["pH"] = df1["pH"]
            df2["DO"] = df1["DO"]
            df1 = df2
        result = df1[["timestamp", "temp", "sal", "turb", "pH", "DO", "nh3"]]
        print(result.tail(5))
        result[["temp", "sal", "turb", "pH", "DO", "nh3"]] = result[["temp", "sal", "turb", "pH", "DO", "nh3"]].clip(self.env.min_space, self.env.max_space)
        result.dropna()
        return result
