import funcoes as f
import config as cf
import cv2
import numpy as np
import os

# Carregar imagem do exame raio-x (já convertido para .png pelo código do Arthur)
exame = cv2.imread(r'C:\Users\julia\OneDrive\Projetos\Priori-RX\nih-dataset\train\Doente\00000077_000.png')

# A imagem .png irá passar pela função que inverte a cor da imagem
imagem = f.preparar_imagem(exame)