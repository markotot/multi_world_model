from src.evaluation.utils import setup_iris, setup_gym, get_checkpoint_path_from_env_id, run_gym_env, plot_images
import torch

if __name__ == "__main__":

    env_id = "BreakoutNoFrameskip-v4"
    #encoder_type = "iris-pre-generated"
    encoder_type = "iris-baseline"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    iris_agent, iris_world_model = setup_iris(env_id, encoder_type, device)
    gym_env = setup_gym(env_id, encoder_type)

    max_steps = 1000

    observations, actions, lost_lives_timesteps = run_gym_env(gym_env, max_steps, iris_agent, device)
    plot_images(observations, current_step=0, num_steps=16, transpose=True, title="Gym Env")
    print("Hello World!")