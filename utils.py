import torch
import torch.nn as nn
import torch.optim as optim
import pygame
from enum import Enum
from pathlib import Path
import numpy as np
import os

BASE=Path("./Imitation")
PATH_MODELO= BASE / "Models"
PATH_DATA = BASE / "Data"
BASE.mkdir(exist_ok=True)
PATH_MODELO.mkdir(exist_ok=True)
PATH_DATA.mkdir(exist_ok=True)


class Red(nn.Module):
    def __init__(self,input_dim:int=128,output_dim:int=5):
        super().__init__()
        capa1=nn.Linear(input_dim,256)
        capa2=nn.Linear(256,128)
        capa3=nn.Linear(128,32)
        capa4=nn.Linear(32,output_dim)

        self.relu = nn.ReLU()

    def forward(self, x:torch.Tensor):
        x = self.relu(self.capa1(x))
        x = self.relu(self.capa2(x))
        x = self.relu(self.capa3(x))
        return self.capa4(x)

def dibujar_frame(pantalla: pygame.Surface, frame: np.ndarray):
    superficie = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
    superficie = pygame.transform.scale(superficie, pantalla.get_size())
    pantalla.blit(superficie, (0, 0))
    pygame.display.flip()

def iniciar_ventana(ancho: int = 480, alto: int = 630) -> pygame.Surface:
    if not pygame.get_init():
        pygame.init()
    pantalla = pygame.display.get_surface()
    if pantalla is None:
        pantalla = pygame.display.set_mode((ancho, alto))
        pygame.display.set_caption("Tetris - Imitation Learning")
    return pantalla

def jugar() -> int:
    if not pygame.display.get_init() or pygame.display.get_surface() is None:
        iniciar_ventana()

    pygame.event.pump()
    #SIN LO DE ARRIBA NO SE VE 
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP] or keys[pygame.K_SPACE]:
        return 1  # ROTAR
    if keys[pygame.K_RIGHT]:
        return 2  # DERECHA
    if keys[pygame.K_LEFT]:
        return 3  # IZQUIERDA
    if keys[pygame.K_DOWN]:
        return 4  # ABAJO
    return 0      # NADA

class Modo(Enum):
    #Los valores son la beta para ver cuantas veces se toma la accion del experto y cuanto del student
    EXPERTO=1
    RED=0
    DAGGER=0.5

def guardar_datos(modelo=None, states=None, actions=None):
    PATH_MODELO.mkdir(parents=True, exist_ok=True)
    PATH_DATA.mkdir(parents=True, exist_ok=True)

    if modelo is not None:
        torch.save(modelo.state_dict(), PATH_MODELO / "modelo.pth")

    if states is not None and actions is not None and len(states) > 0:
        np.savez_compressed(
            PATH_DATA / "data.npz",
            states=np.array(states, dtype=np.float32),
            actions=np.array(actions, dtype=np.int64),
        )


def cargar_datos(modelo,device,path_modelo: Path = PATH_MODELO / "modelo.pth", path_datos: Path = PATH_DATA / "data.npz") -> tuple[list, list]:
    states, actions = [], []

    if path_modelo.exists():
        modelo.load_state_dict(torch.load(path_modelo, map_location=device, weights_only=True))

    if path_datos.exists():
        datos = np.load(path_datos)
        states = list(datos["states"])
        actions = datos["actions"].tolist()

    return states, actions