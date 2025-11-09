from gsheet import StateData
from stable_baselines3 import TD3
import pandas as pd

def aksi_temp_ket(aksi_temp):
    if aksi_temp < 0:
        return "Matikan Pemanas"
    return "Hidupkan Pemanas"

def aksi_sal_ket(aksi_sal):
    if aksi_sal < 0:
        return "Tambah Air Tawar"
    return "Tambah Air Asin"

def aksi_turb_ket(aksi_turb):
    if aksi_turb < 0:
        return "Tambah Air Bersih"
    return "Tambah Air Keruh"

def aksi_pH_ket(aksi_pH):
    if aksi_pH < 0:
        return "Tambah Zat Basa"
    return "Tambah Zat Asam"

def aksi_DO_ket(aksi_DO):
    if aksi_DO < 0:
        return "Matikan Aerator"
    return "Hidupkan Aerator"

def aksi_nh3_ket(aksi_nh3):
    if aksi_nh3 < 0:
        return "Aerasi/Nitrifikasi"
    return "Tambah Beban Organik"

def tingkat_rekom(aksi):
    aksi = abs(aksi)
    if aksi > 2/3:
        return "tinggi"
    elif aksi > 1/3:
        return "sedang"
    else:
        return "rendah"

td3 = TD3.load("td3_water2.zip")
sheet = StateData()
data = sheet.read()
state = (data.values[:, 1:] - sheet.env.min_space) / sheet.env.range_space
actions, _ = td3.predict(state)
actions_data = pd.DataFrame(actions, columns=["aksi_temp", "aksi_sal", "aksi_turb", "aksi_pH", "aksi_DO", "aksi_nh3"])

actions_data["aksi_temp ket"] = actions_data["aksi_temp"].apply(aksi_temp_ket)
actions_data["aksi_sal ket"] = actions_data["aksi_sal"].apply(aksi_sal_ket)
actions_data["aksi_turb ket"] = actions_data["aksi_turb"].apply(aksi_turb_ket)
actions_data["aksi_pH ket"] = actions_data["aksi_pH"].apply(aksi_pH_ket)
actions_data["aksi_DO ket"] = actions_data["aksi_DO"].apply(aksi_DO_ket)
actions_data["aksi_nh3 ket"] = actions_data["aksi_nh3"].apply(aksi_nh3_ket)

for col in ["aksi_temp", "aksi_sal", "aksi_turb", "aksi_pH", "aksi_DO", "aksi_nh3"]:
    actions_data[f"{col} rekom"] = actions_data[col].apply(tingkat_rekom)

result = pd.concat([data, actions_data], axis=1)
print(result.tail())
result.to_excel("extracted.xlsx")
