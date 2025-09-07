from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from stable_baselines3 import TD3
from water_env import WaterEnv
from fastapi.middleware.cors import CORSMiddleware
from gsheet import StateData
import pandas as pd


class AppConfig:
    def __init__(self) -> None:
        self.state = StateData()
        self.td3 = TD3.load("td3_water2.zip")
        self.env = WaterEnv()


app_config = AppConfig()
app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173",
    "http://localhost:4173",
    "https://research1-46folkjlkjlk.vercel.app",
    "https://www.research-td3-shrimp.com",
    "https://research-td3-shrimp.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/info")
def info():
    return {
        "min_space": app_config.env.min_space.tolist(),
        "max_space": app_config.env.max_space.tolist(),
        "target_space": app_config.env.target_space.tolist(),
    }


@app.get("/states")
def index():
    states = app_config.state.read().tail(50)
    return JSONResponse(states.to_dict("records"))


@app.post("/actions")
async def action(req: Request):
    data = await req.json()
    df = pd.DataFrame(data)[["timestamp", "temp", "sal", "turb", "pH", "DO", "nh3"]]
    state = (df.values[:, 1:] - app_config.env.min_space) / app_config.env.range_space
    action, _ = app_config.td3.predict(state)
    result = pd.DataFrame(action, columns=["temp", "sal", "turb", "pH", "DO", "nh3"])
    return JSONResponse(result.to_dict("records"))
