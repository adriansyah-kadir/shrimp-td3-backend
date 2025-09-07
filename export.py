from stable_baselines3 import TD3
import torch as th

td3 = TD3.load("td3_water2.zip")

th.onnx.export(
    td3.policy,
    (th.randn((1, 6)),),
    "td3_water.onnx",
    input_names=["states"],
    output_names=["actions"],
)
