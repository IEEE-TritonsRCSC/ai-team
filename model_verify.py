import torch
ckpt = torch.load("models/discrete_ppo_policy_ep1000.pth", map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt)
if "encoder.0.weight" in sd:
    print(sd["encoder.0.weight"].shape[1])