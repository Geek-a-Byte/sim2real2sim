import torch
from stable_baselines3 import PPO

def export_ppo_to_onnx(model_path, onnx_path, obs_shape):
    """
    Exports a Stable-Baselines3 PPO policy to ONNX format.
    """
    model = PPO.load(model_path)
    policy = model.policy

    class OnnxablePolicy(torch.nn.Module):
        def __init__(self, extractor, action_net, value_net):
            super().__init__()
            self.extractor = extractor
            self.action_net = action_net
            self.value_net = value_net

        def forward(self, observation):
            # Extract features and pass through the action network
            features = self.extractor(observation)
            return self.action_net(features)

    onnxable_model = OnnxablePolicy(
        policy.features_extractor,
        policy.action_net,
        policy.value_net
    )

    dummy_input = torch.randn(1, *obs_shape)
    
    torch.onnx.export(
        onnxable_model,
        dummy_input,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["action"]
    )
    print(f"Model successfully exported to {onnx_path}")

if __name__ == "__main__":
    # Example export for Phase 1 (Goalkeeper)
    # Obs shape is (6,) based on M2
    export_ppo_to_onnx("logs/gk_final.zip", "deploy/gk_policy.onnx", obs_shape=(6,))