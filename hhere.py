
# -*- coding: utf-8 -*-
"""
Travail pratique - Segmentation et extraction d'information
Fichier Python répondant point par point au sujet donné.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_bool
from skimage.morphology import binary_erosion, binary_dilation, binary_opening, binary_closing


image_path=""

# Question 2 - Histogramme
def question2_histogramme(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    plt.figure()
    plt.title("Q2 - Histogramme de l'image en niveau de gris")
    plt.xlabel("Niveaux de gris")
    plt.ylabel("Nombre de pixels")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.grid()
    plt.show()

# Question 4-5 - Binarisation manuelle avec seuil fixé
def question4_5_binarisation_man(image_path, seuil=128):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, seuil, 255, cv2.THRESH_BINARY)

    cv2.imshow("Q4 - Image originale", gray)
    cv2.imshow("Q5 - Binarisation manuelle", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return binary

# Question 6 - Binarisation automatique (méthode d'Otsu)
def question6_binarisation_auto(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    seuil, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow("Q6 - Image originale", gray)
    cv2.imshow("Q6 - Binarisation auto (Otsu)", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"Seuil trouvé par Otsu : {seuil}")
    return binary

# Question 8 - Affichage de l'histogramme de l'image binaire
def question8_hist_binaire(binaire):
    hist = cv2.calcHist([binaire], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Q8 - Histogramme de l'image binaire")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.grid()
    plt.show()

# Question 11 - Détection de forme binaire
def question11_binarisation_forme(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    obj = (binary == 255).astype(np.uint8)
    return obj

# Question 12 à 14 - Ouverture image + transformation en niveaux de gris + histogramme
def question12_13_14_process(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Q13 - Image en niveaux de gris", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Q14 - Histogramme monochrome")
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.grid()
    plt.show()

# Question 15 - Binarisation multiple avec opérateurs logiques
def question15_binarisation_double(image_path, seuil1=100, seuil2=200):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    bin1 = img > seuil1
    bin2 = img < seuil2
    combined = np.logical_and(bin1, bin2).astype(np.uint8) * 255
    return combined

# Question 16 - Nettoyage avec imclearborder équivalent
from skimage.segmentation import clear_border
def question16_clear_border(binary_image):
    cleared = clear_border(binary_image)
    return cleared

# Question 17 - Morphologie : érosion, dilatation, ouverture, fermeture
def question17_operations_morpho(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
    binary_bool = img_as_bool(binary)

    erosion = binary_erosion(binary_bool)
    dilation = binary_dilation(binary_bool)
    opening = binary_opening(binary_bool)
    closing = binary_closing(binary_bool)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    titles = ['Original', 'Érosion', 'Dilatation', 'Ouverture', 'Fermeture']
    images = [binary_bool, erosion, dilation, opening, closing]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
