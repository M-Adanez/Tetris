import gymnasium as gym
import ale_py
import random
from utils import *

gym.register_envs(ale_py)

BASE.mkdir(exist_ok=True)
PATH_MODELO.mkdir(exist_ok=True)
PATH_DATA.mkdir(exist_ok=True)

SEED = random.randint(0, 2**32 - 1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env=gym.make("ALE/Tetris-v5",render_mode="rgb_array",obs_type="ram",)

state,_=env.reset(seed=SEED)

modo=Modo.EXPERTO

modelo=Red(input_dim=128,output_dim=5)
states_df,actions_df=cargar_datos(modelo,device)

clock = pygame.time.Clock()
if modo==Modo.EXPERTO:
    partidas=3 #Asi tiene varios inicios

    for _ in range(partidas):
        state, _ = env.reset(seed=SEED)
        for _ in range(1000):#100 acciones por partida
            dibujar_frame(iniciar_ventana(), env.render())
            action=jugar()

            states_df.append(state.astype(np.float32))
            actions_df.append(action)
            
            state, reward, terminated, truncated, info = env.step(action)
            clock.tick(10)#SIN ESTO VA A TODA LECHE

    guardar_datos(modelo,states_df,actions_df)



