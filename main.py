import gymnasium as gym
import ale_py
import random
from utils import *

gym.register_envs(ale_py)

BASE.mkdir(exist_ok=True)
PATH_MODELO.mkdir(exist_ok=True)
PATH_DATA.mkdir(exist_ok=True)

SEED = random.randint(0, 2**32 - 1)
PORCENTAJE_GUARDADO=0.2
PARTIDAS=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env=gym.make("ALE/Tetris-v5",render_mode="rgb_array",obs_type="ram",)

# state,_=env.reset(seed=SEED)

modo=Modo.RED

modelo=Red(input_dim=128,output_dim=5).to(device)
states,actions=cargar_datos(modelo,device)

clock = pygame.time.Clock()
if modo==Modo.EXPERTO:
    #Asi tiene varios inicios para empezar

    for p in range(PARTIDAS):

        state, _ = env.reset()

        for _ in range(1000):#100 acciones por partida
            dibujar_frame(iniciar_ventana(), env.render())
            action=jugar()

            if action != 0 or random.random() < PORCENTAJE_GUARDADO: #gaurdar solo algunas acciones que sean menores que 0
                states.append(state.astype(np.float32) / 255.0)#normalizar estado de imagen
                actions.append(action)
                pass
            
            state, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
            clock.tick(10) #SIN ESTO VA A TODA LECHE


    print("GUARDANDO DATOS")
    guardar_datos(modelo,states,actions)
    print("DATOS GUARDADOS")

if modo==Modo.RED:
    modelo=entrenar_modelo(modelo,states,actions,device)
    guardar_datos(modelo=modelo)
    for p in range(PARTIDAS):
            SEED = random.randint(0, 2**32 - 1)
            state, _ = env.reset(seed=SEED)
            for _ in range(1000):#100 acciones por partida
                modelo.eval()
                with torch.no_grad():
                    st=state.astype(np.float32)/255
                    x = torch.from_numpy(st).unsqueeze(0).to(device)
                    accion_red = int(torch.argmax(modelo(x), dim=-1).item())

                dibujar_frame(iniciar_ventana(), env.render())
                
                state, reward, terminated, truncated, info = env.step(accion_red)
                if terminated or truncated:
                    break
                # clock.tick(5) #SIN ESTO VA A TODA LECHE
                print("SCORE:",reward)

