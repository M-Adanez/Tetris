# from utils import *
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# modelo=Red(input_dim=128,output_dim=5).to(device)

# states_df,actions_df=cargar_datos(modelo,device)
# print(states_df[0])
# print(actions_df[0])


import gymnasium as gym, ale_py, numpy as np

gym.register_envs(ale_py)
env = gym.make("ALE/Tetris-v5", obs_type="ram", repeat_action_probability=0.0)

ram_pieza_1, _ = env.reset()

# Bajamos pulsando ABAJO (4) hasta que la 1ª pieza aterrice y aparezca la 2ª
for paso in range(200):
    ram_actual, _, _, _, _ = env.step(4)
    # Al nacer una pieza nueva arriba, cambian de golpe los bytes de estado de pieza
    if paso > 20 and not np.array_equal(ram_pieza_1[90:], ram_actual[90:]):
        diff = np.where(ram_pieza_1 != ram_actual)[0]
        print(f"Paso {paso} | Índices que cambiaron: {diff}")
        for idx in diff:
            if idx >= 90:
                print(f"  ram[{idx}]: {ram_pieza_1[idx]} -> {ram_actual[idx]}")
        break