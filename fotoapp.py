import requests
from PIL import Image, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
from skimage import img_as_float, img_as_ubyte
from io import BytesIO

dimensiones_redes = {
    "Youtube": (1280, 720),
    "Instagram": (1080, 1080),
    "Twitter": (1200, 675),
    "Facebook": (1200, 630)
}

def redimensionar_imagen(imagen, app):
    if app not in dimensiones_redes:
        raise ValueError(f"La red social a la cual quieres adaptar la imagen no es válida. Elija entre: 'Youtube', 'Instagram', 'Twitter', 'Facebook'")

    ancho, altura = dimensiones_redes[app]

    if imagen.startswith("http"):
        try:
            response = requests.get(imagen)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            raise ValueError(f"No se pudo descargar la imagen desde la URL. Error: {e}")
    else:
        try:
            img = Image.open(imagen)
        except Exception as e:
            raise ValueError(f"No se pudo abrir la imagen. Error: {e}")

    if img is None:
        raise ValueError("La imagen no se pudo cargar correctamente.")

    ancho_original, altura_original = img.size

    if altura_original == 0:
        raise ValueError("La altura de la imagen original es cero, lo que no es válido.")
    
    radio = ancho_original / altura_original
    if ancho / altura > radio:
        nueva_altura = altura
        nuevo_ancho = int(radio * nueva_altura)
    else:
        nuevo_ancho = ancho
        nueva_altura = int(nuevo_ancho / radio)

    imagen_redimensionada = img.resize((nuevo_ancho, nueva_altura), Image.LANCZOS)

    final_img = Image.new("RGB", (ancho, altura), (255, 255, 255))

    final_img.paste(imagen_redimensionada, ((ancho - nuevo_ancho) // 2, (altura - nueva_altura) // 2))

    return final_img

def ajustar_contraste(imagen):
    img = imagen.convert('L')

    img_array = np.array(img)
    
    img_eq = exposure.equalize_hist(img_as_float(img_array))
    
    img_eq = img_as_ubyte(img_eq)
    
    img_eq = Image.fromarray(img_eq)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Imagen Original")
    axes[0].axis('off')  # Quitar los ejes
    
    axes[1].imshow(img_eq, cmap='gray')
    axes[1].set_title("Imagen Ecualizada")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

    img_eq.save('imagen_ecualizada.jpg')
    img.save('imagen_original.jpg')

def aplicar_filtro(imagen_redimensionada, filtro_elegido):
    filtros = {
        "ORIGINAL": None,
        "BLUR": ImageFilter.BLUR,
        "CONTOUR": ImageFilter.CONTOUR,
        "DETAIL": ImageFilter.DETAIL,
        "EDGE ENHANCE": ImageFilter.EDGE_ENHANCE,
        "EDGE ENHANCE MORE": ImageFilter.EDGE_ENHANCE_MORE,
        "EMBOSS": ImageFilter.EMBOSS,
        "FIND EDGES": ImageFilter.FIND_EDGES,
        "SHARPEN": ImageFilter.SHARPEN,
        "SMOOTH": ImageFilter.SMOOTH
    }

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()

    axes[0].imshow(imagen_redimensionada)
    axes[0].set_title("ORIGINAL", fontsize=14)
    axes[0].axis('off')

    for idx, (nombre_filtro, filtro) in enumerate(filtros.items(), start=1):
        if filtro is None:
            img_filtrada = imagen_redimensionada
        else:
            img_filtrada = imagen_redimensionada.filter(filtro)
        
        axes[idx].imshow(img_filtrada)
        axes[idx].set_title(nombre_filtro, fontsize=14, color="red" if nombre_filtro == filtro_elegido else "black")
        axes[idx].axis('off')

        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
        axes[idx].set_xticklabels([])
        axes[idx].set_yticklabels([])

    plt.subplots_adjust(wspace=0, hspace=0)

    for i in range(len(filtros), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    if filtro_elegido in filtros and filtros[filtro_elegido] is not None:
        img_filtrada = imagen_redimensionada.filter(filtros[filtro_elegido])
        img_filtrada.save(f'{filtro_elegido}_imagen_resultante.jpg')
        print(f'Imagen con filtro "{filtro_elegido}" guardada como "{filtro_elegido}_imagen_resultante.jpg".')
    else:
        imagen_redimensionada.save(f'ORIGINAL_imagen_resultante.jpg')
        print('Imagen original guardada como "ORIGINAL_imagen_resultante.jpg".')

def procesar_boceto(imagen_redimensionada):
    img_cv = cv2.cvtColor(np.array(imagen_redimensionada), cv2.COLOR_RGB2BGR)

    gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)

    sobel = cv2.magnitude(sobel_x, sobel_y)

    _, sobel_binarizado = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)

    sobel_binarizado_pil = Image.fromarray(sobel_binarizado)

    sobel_binarizado_pil = sobel_binarizado_pil.convert("1")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(imagen_redimensionada)
    axes[0].set_title("Imagen Original")
    axes[0].axis('off')

    axes[1].imshow(sobel_binarizado_pil, cmap='gray')
    axes[1].set_title("Bordes Sobel Binarizados")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    sobel_binarizado_pil.save("bordes_sobel_binarizados.jpg")
    
    print("Imagen con bordes Sobel binarizados guardada como 'bordes_sobel_binarizados.jpg'.")