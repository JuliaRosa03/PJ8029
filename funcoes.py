import cv2
import config as cf

#>>>>>> Será substituida pela versão do arthur
# Função para carregar e pre-processar a imagem - converter para array 
def preparar_imagem(imagem):
    imagem_redimensionada = cv2.resize(imagem, cf.tamanho)
    # Inverter as cores da imagem
    imagem_invertida = cv2.bitwise_not(imagem_redimensionada)

    return imagem_invertida
    
# Função para carregar e pre-processar a imagem - converter para array
def carregar_e_processar_imagem(img_path):
    img = image.load_img(img_path, target_size=cf.tamanho)  # Altere o tamanho conforme necessário
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# >>>>>>>>>>>>>>>>>>>>

def analise_da_imagem(img_array):
    # Carregar o modelo
    modelo = load_model(r"C:\Users\julia\OneDrive\Projetos\Priori-RX\teste\CNN_RaioX_COVID.weights.h5", compile=False)
    # Fazer a previsão
    predictions = modelo.predict(img_array)
    # Se você souber as classes, pode imprimir a classe prevista
    predicted_class = np.argmax(predictions, axis=1)

    if predicted_class == 0:
        classe = 'Doente'
    else:
        classe = 'Saudavel'

    return classe