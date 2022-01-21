import cv2 as cv
import numpy as np
import math
import os
from scipy.stats import entropy

def create_merkmale(pfad):
    path, dirs, files = next(os.walk("bilder/" + str(pfad) + "_neu/"))
    file_count = len(files)
    anz = 0
    # Ausgabevariablen
    Flaeche_R = []
    Flaeche_C = []
    Umfang_R = []
    Umfang_C = []
    seitenverh_list = []
    verh_flaeche_umfang_r_list = []
    anz_kanten_list = []
    anz_corners_list = []
    hierarchy_len = []

    # kernel für dilate
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))

    for anz in range(file_count):

        # Eingabe
        img = cv.imread("bilder/" + str(pfad) + "_neu/" + str(pfad) + " (" + str(
            anz) + ").jpg")  # hier checken ob quadratisch hoch oder querformat (4:3)

        # Bearbeiten
        # img_re=cv.resize(img,(255,255)) #wenn 4:3 ansonsten quadratisch
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # blur hilft bei einzelnen kontrastreichen pixeln oder strukturen im hintergrund darf nur ungerade sein
        img_b = cv.medianBlur(img_gray, 7)

        # canny
        img_canny = cv.Canny(img_b, 10, 200)
        anz_kanten = np.count_nonzero(img_canny)
        anz_kanten_list = np.append(anz_kanten_list, anz_kanten)

        # adaptive gaussian thresholding + invertieren
        """
        img_tgauss = cv.adaptiveThreshold(img_b,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        img_tgauss_n = ~img_tgauss
        """

        # dilate um Kanten dicker zu machen
        img_canny_dil = cv.dilate(img_canny, kernel)

        # shi-tomasi corner detection

        corners = cv.goodFeaturesToTrack(img_b, 80, 0.03, 10)
        anz_corners_list = np.append(anz_corners_list, len(corners))

        # corner anzeigen
        """
        for i in corners:
            x, y = i.ravel()
            cv.circle(img, (x, y), 3, 255, -1)
        plt.imshow(img), plt.show()
        """
        # contours und flächen
        contours, hierarchy = cv.findContours(img_canny_dil, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        x = len(contours) - 1
        # print("Anzahl Konturen: ",x+1)
        i = 0
        area = []
        area = [0 for n in range(x + 1)]
        while i <= x:
            cnt = contours[i]
            area[i] = cv.contourArea(cnt)
            img_conturs = cv.drawContours(img, [cnt], 0, (255, 0, 0), 3)
            i = i + 1

        # größte Fläche finden
        max_area = max(area)
        pos_max_area = [p for p, q in enumerate(area) if q == max_area]

        # print(hierarchy[0])
        # print(len(hierarchy[0]))
        hierarchy_len = np.append(hierarchy_len, len(hierarchy[0]))
        # print("Fläche Kontur:",area[pos_max_area[0]])

        # Verhältnis Fläche zu Umfang von größter Fläche
        umfang = cv.arcLength(contours[pos_max_area[0]], True)
        Umfang_C = np.append(Umfang_C, umfang)
        # print("Umfang:",umfang)
        # verh_flaeche_umfang = area[pos_max_area[0]]/umfang
        # print("Fläche durch Umfang:",verh_flaeche_umfang)

        # rotated rectangle zeichnen und contour mit größter fläche

        cntr = contours[pos_max_area[0]]
        rect = cv.minAreaRect(cntr)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        img_rect = cv.drawContours(img, [box], 0, (0, 0, 255), 2)

        # Fläche und Verhältnis Fläche zu Umfang von größter Fläche Rechteck
        flaeche_r = cv.contourArea(box)
        Flaeche_R = np.append(Flaeche_R, flaeche_r)

        umfang_r = cv.arcLength(box, True)
        Umfang_R = np.append(Umfang_R, umfang_r)

        verh_flaeche_umfang_r = flaeche_r / umfang_r
        verh_flaeche_umfang_r_list = np.append(verh_flaeche_umfang_r_list, verh_flaeche_umfang_r)

        # Verhätnis der Seiten
        seite1 = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        seite2 = math.sqrt((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2)

        if seite1 >= seite2:
            seitenverh = seite1 / seite2
        else:
            seitenverh = seite2 / seite1

        seitenverh_list = np.append(seitenverh_list, seitenverh)

        Flaeche_C = np.append(Flaeche_C, area[pos_max_area[0]]/(seite1*seite2))

        # jedes Bild anzeigen
        """
        cv.imshow("unskaliert", img)
        cv.imshow("gauss", img_tgauss_n)
        cv.imshow("blur", img_b)
        cv.imshow("dilation", img_tgauss_n_dil)
        k = cv.waitKey(0)
        """
    return (seitenverh_list, Flaeche_C, anz_kanten_list)

def testbild_merkmale():
    path, dirs, files = next(os.walk("bilder/testbilder"))
    file_count = len(files)
    anz = 0
    # Ausgabevariablen
    Flaeche_R = []
    Flaeche_C = []
    Umfang_R = []
    Umfang_C = []
    seitenverh_list = []
    verh_flaeche_umfang_r_list = []
    anz_kanten_list = []
    anz_corners_list = []
    hierarchy_len = []

    # kernel für dilate
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5), (-1, -1))

    for anz in range(1):

        # Eingabe
        img = cv.imread("bilder/testbilder/tb_0.jpg")  # hier checken ob quadratisch hoch oder querformat (4:3)

        # Bearbeiten
        # img_re=cv.resize(img,(255,255)) #wenn 4:3 ansonsten quadratisch
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # blur hilft bei einzelnen kontrastreichen pixeln oder strukturen im hintergrund darf nur ungerade sein
        img_b = cv.medianBlur(img_gray, 7)

        # canny
        img_canny = cv.Canny(img_b, 10, 200)
        anz_kanten = np.count_nonzero(img_canny)
        anz_kanten_list = np.append(anz_kanten_list, anz_kanten)

        # adaptive gaussian thresholding + invertieren
        """
        img_tgauss = cv.adaptiveThreshold(img_b,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
        img_tgauss_n = ~img_tgauss
        """

        # dilate um Kanten dicker zu machen
        img_canny_dil = cv.dilate(img_canny, kernel)

        # shi-tomasi corner detection

        corners = cv.goodFeaturesToTrack(img_b, 80, 0.03, 10)
        anz_corners_list = np.append(anz_corners_list, len(corners))

        # corner anzeigen
        """
        for i in corners:
            x, y = i.ravel()
            cv.circle(img, (x, y), 3, 255, -1)
        plt.imshow(img), plt.show()
        """
        # contours und flächen
        contours, hierarchy = cv.findContours(img_canny_dil, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        x = len(contours) - 1
        # print("Anzahl Konturen: ",x+1)
        i = 0
        area = []
        area = [0 for n in range(x + 1)]
        while i <= x:
            cnt = contours[i]
            area[i] = cv.contourArea(cnt)
            img_conturs = cv.drawContours(img, [cnt], 0, (255, 0, 0), 3)
            i = i + 1

        # größte Fläche finden
        max_area = max(area)
        pos_max_area = [p for p, q in enumerate(area) if q == max_area]

        # print(hierarchy[0])
        # print(len(hierarchy[0]))
        hierarchy_len = np.append(hierarchy_len, len(hierarchy[0]))
        # print("Fläche Kontur:",area[pos_max_area[0]])

        # Verhältnis Fläche zu Umfang von größter Fläche
        umfang = cv.arcLength(contours[pos_max_area[0]], True)
        Umfang_C = np.append(Umfang_C, umfang)
        # print("Umfang:",umfang)
        # verh_flaeche_umfang = area[pos_max_area[0]]/umfang
        # print("Fläche durch Umfang:",verh_flaeche_umfang)

        # rotated rectangle zeichnen und contour mit größter fläche

        cntr = contours[pos_max_area[0]]
        rect = cv.minAreaRect(cntr)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        img_rect = cv.drawContours(img, [box], 0, (0, 0, 255), 2)

        # Fläche und Verhältnis Fläche zu Umfang von größter Fläche Rechteck
        flaeche_r = cv.contourArea(box)
        Flaeche_R = np.append(Flaeche_R, flaeche_r)

        umfang_r = cv.arcLength(box, True)
        Umfang_R = np.append(Umfang_R, umfang_r)

        verh_flaeche_umfang_r = flaeche_r / umfang_r
        verh_flaeche_umfang_r_list = np.append(verh_flaeche_umfang_r_list, verh_flaeche_umfang_r)

        # Verhätnis der Seiten
        seite1 = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        seite2 = math.sqrt((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2)

        if seite1 >= seite2:
            seitenverh = seite1 / seite2
        else:
            seitenverh = seite2 / seite1

        seitenverh_list = np.append(seitenverh_list, seitenverh)

        Flaeche_C = np.append(Flaeche_C, area[pos_max_area[0]] / (seite1 * seite2))

        # jedes Bild anzeigen
        cv.imshow("Testbild", img)
        k = cv.waitKey(0)
        """
        cv.imshow("gauss", img_tgauss_n)
        cv.imshow("blur", img_b)
        """
    return (seitenverh_list, Flaeche_C, anz_kanten_list)

def kategorisieren(merkmal_array_1, merkmal_array_2, testbild_array, steps):
    array_1_kat = []
    array_2_kat = []
    testbild_kat = []
    merkmal_array_max = np.maximum(np.max(merkmal_array_1), np.max(merkmal_array_2)) # maximalen Merkmalswert über beide Arrays ermitteln
    merkmal_array_min = np.minimum(np.min(merkmal_array_1), np.min(merkmal_array_2)) # minimalen Merkmalswert über beide Arrays ermitteln
    # steps = round((round(merkmal_array_max + 0.5) - round(merkmal_array_min - 0.5)) / intervall) # Anzahl Schritte über auf-/abgerundeten Merkmalswert ermitteln
    intervallpunkte = np.linspace(merkmal_array_min, merkmal_array_max, num=steps, retstep=True)
    # print(intervallpunkte)
    intervall=intervallpunkte[1]
    # print(intervall)

    for i in range(len(merkmal_array_1)): # Schleife geht alle Bilder in diesem Array durch
        untere_grenze = merkmal_array_min
        obere_grenze = merkmal_array_min + intervall
        for j in range(steps): # Schleife geht alle Kategorien durch und checkt ob der Wert des Bilds in der jeweiligen Kategorie ist
            if merkmal_array_1[i] >= untere_grenze and merkmal_array_1[i] < obere_grenze:
                array_1_kat = np.append(array_1_kat, j)
            untere_grenze = untere_grenze + intervall
            obere_grenze = obere_grenze + intervall

    for i in range(len(merkmal_array_2)):
        untere_grenze = merkmal_array_min
        obere_grenze = merkmal_array_min + intervall
        for j in range(steps):
            if merkmal_array_2[i] >= untere_grenze and merkmal_array_2[i] < obere_grenze:
                array_2_kat = np.append(array_2_kat, j)
            untere_grenze = untere_grenze + intervall
            obere_grenze = obere_grenze + intervall


    for i in range(len(testbild_array)): # die selbe Schleife wie oben, nur für Testbilder
        untere_grenze = merkmal_array_min
        obere_grenze = merkmal_array_min + intervall
        if testbild_array[i] < untere_grenze:
            testbild_kat = np.append(testbild_kat, 0)
        for j in range(steps):
            if testbild_array[i] >= untere_grenze and testbild_array[i] < obere_grenze:
                testbild_kat = np.append(testbild_kat, j)
            untere_grenze = untere_grenze + intervall
            obere_grenze = obere_grenze + intervall
        if testbild_array[i] >= obere_grenze:
            testbild_kat = np.append(testbild_kat, (steps-2))

    return (array_1_kat, array_2_kat, testbild_kat) #Rückgabe der kategorisierten Werte ans Hauptprogramm

def bayes_merkmale(array_locher_kat, array_tacker_kat, array_testbild_kat):
    # Definition der a-priori Wahrscheinlichkeiten
    a_priori_tacker = (len(array_tacker_kat)) / (len(array_tacker_kat) + len(array_locher_kat))
    a_priori_locher = (len(array_locher_kat)) / (len(array_tacker_kat) + len(array_locher_kat))

    # Kategorie x von der die Wahrscheinlichkeit gesucht werden soll (aus Testbild-Array genommen)
    x = round(array_testbild_kat[0])
    # print(x)

    i = 0
    zaehler_tacker = 0
    # Zählen wie oft Kategorie x in Array vom Tacker vorkommt
    for i in range(len(array_tacker_kat)):
        if round(array_tacker_kat[i]) == x:
            zaehler_tacker = zaehler_tacker + 1

    # print(zaehler_tacker)
    # print(len(array_tacker_kat))
    # prozentuale Wahrscheinlichkeit von x in Tacker
    x_tacker = zaehler_tacker / len(array_tacker_kat)
    # print(x_tacker)

    j = 0
    zaehler_locher = 0
    # Zählen wie oft Kategorie x in Array von Locher vorkommt
    for j in range(len(array_locher_kat)):
        if round(array_locher_kat[j]) == x:
            zaehler_locher = zaehler_locher + 1

    # print(zaehler_locher)
    # print(len(array_locher_kat))
    # prozentuale Wahrscheinlichkeit von x in Locher
    x_locher = zaehler_locher / len(array_locher_kat)
    # print(x_locher)

    # Bestimmung der Wahrscheinlichkeit nach Bayes-Formel
    bayes_tacker_x = (x_tacker * a_priori_tacker) / (x_tacker * a_priori_tacker + x_locher * a_priori_locher)
    bayes_locher_x = (x_locher * a_priori_locher) / (x_tacker * a_priori_tacker + x_locher * a_priori_locher)

    return (bayes_locher_x, bayes_tacker_x)

"""
===================================================================================================================
Hauptprogramm
===================================================================================================================
"""

# Arrays deklarieren
seitenverh_tacker = []
seitenverh_tacker_kat = []
seitenverh_locher = []
seitenverh_locher_kat = []
seitenverh_testbild= []
seitenverh_testbild_kat = []

flaeche_tacker = []
flaeche_tacker_kat = []
flaeche_locher = []
flaeche_locher_kat = []
flaeche_testbild= []
flaeche_testbild_kat = []

kanten_tacker = []
kanten_tacker_kat = []
kanten_locher = []
kanten_locher_kat = []
kanten_testbild= []
kanten_testbild_kat = []

# Seitenverhältnis-Arrays einlesen und Merkmal zum Check-Up ausgeben
seitenverh_locher, flaeche_locher, kanten_locher = create_merkmale("Locher")
print("Locher anzahl:", len(seitenverh_locher))
print("Locher mittelwert seitenverhältnis:", np.mean(seitenverh_locher))
print("Locher standardabw. gesamtmenge seitenverhältnis:", np.std(seitenverh_locher))
print("Locher maximales seitenverhältnis:", np.max(seitenverh_locher))
print("Locher mittelwert Fläche:", np.mean(flaeche_locher))
print("Locher standardabw. gesamtmenge Fläche:", np.std(flaeche_locher))
print("Locher maximale Fläche:", np.max(flaeche_locher))
print("Locher minimale Fläche:", np.min(flaeche_locher))
print("Locher mittelwert Kanten:", np.mean(kanten_locher))
print("Locher standardabw. gesamtmenge Kanten:", np.std(kanten_locher))
print("Locher maximale Kanten:", np.max(kanten_locher))
print("Locher minimale Kanten:", np.min(kanten_locher))
print("-----------------------------------------------------------------------------------------")
seitenverh_tacker, flaeche_tacker, kanten_tacker = create_merkmale("Tacker")
print("Tacker anzahl:", len(seitenverh_tacker))
print("Tacker mittelwert seitenverhältnis:", np.mean(seitenverh_tacker))
print("Tacker standardabw. gesamtmenge seitenverhältnis:", np.std(seitenverh_tacker))
print("Tacker maximales seitenverhältnis:", np.max(seitenverh_tacker))
print("Tacker mittelwert Fläche:", np.mean(flaeche_tacker))
print("Tacker standardabw. gesamtmenge Fläche:", np.std(flaeche_tacker))
print("Tacker maximale Fläche:", np.max(flaeche_tacker))
print("Tacker minimale Fläche:", np.min(flaeche_tacker))
print("Tacker mittelwert Kanten:", np.mean(kanten_tacker))
print("Tacker standardabw. gesamtmenge Kanten:", np.std(kanten_tacker))
print("Tacker maximale Kanten:", np.max(kanten_tacker))
print("Tacker minimale Kanten:", np.min(kanten_tacker))
print("-----------------------------------------------------------------------------------------")

# Testbild-Arrays einlesen und Merkmal zum Check-Up ausgeben
seitenverh_testbild, flaeche_testbild, kanten_testbild = testbild_merkmale()
print("Testbild anzahl:", len(seitenverh_testbild))
print("Testbild Seitenverhältnis:", seitenverh_testbild)
print("Testbild Fläche:", flaeche_testbild)
print("Testbild Kanten:", kanten_testbild)
print("-----------------------------------------------------------------------------------------")

# Seitenverhältnis-Arrays von Trainings- und Testbildern kategorisieren
anz_kategorien = 10
seitenverh_locher_kat, seitenverh_tacker_kat, seitenverh_testbild_kat=kategorisieren(seitenverh_locher, seitenverh_tacker, seitenverh_testbild, anz_kategorien)

"""
print(seitenverh_tacker_kat)
print(len(seitenverh_tacker_kat))
print(seitenverh_tacker)
print(len(seitenverh_tacker))

print(seitenverh_locher_kat)
print(len(seitenverh_locher_kat))
print(seitenverh_locher)
print(len(seitenverh_locher))

print(seitenverh_testbild_kat)
print(len(seitenverh_testbild_kat))
print(seitenverh_testbild)
print(len(seitenverh_testbild))
"""

flaeche_locher_kat, flaeche_tacker_kat, flaeche_testbild_kat=kategorisieren(flaeche_locher, flaeche_tacker, flaeche_testbild, anz_kategorien)

kanten_locher_kat, kanten_tacker_kat, kanten_testbild_kat=kategorisieren(kanten_locher, kanten_tacker, kanten_testbild, anz_kategorien)

"""
====================================================================================================================
Hauptprogramm: Bayes
====================================================================================================================
"""

seitenverh_bayes_locher, seitenverh_bayes_tacker = bayes_merkmale(seitenverh_locher_kat, seitenverh_tacker_kat, seitenverh_testbild_kat)
if seitenverh_bayes_locher <= 0.45:
    print("Tacker erkannt.")
elif seitenverh_bayes_locher >= 0.55:
    print("Locher erkannt.")
else:
    print("Weder Locher noch Tacker erkannt.")

print("Bayes-Wahrscheinlichkeit Seitenverhältnis Locher: " + str(seitenverh_bayes_locher))
print("Bayes-Wahrscheinlichkeit Seitenverhältnis Tacker: " + str(seitenverh_bayes_tacker))
print("-----------------------------------------------------------------------------------------")

flaeche_bayes_locher, flaeche_bayes_tacker = bayes_merkmale(flaeche_locher_kat, flaeche_tacker_kat, flaeche_testbild_kat)
print("Bayes-Wahrscheinlichkeit Fläche Locher: " + str(flaeche_bayes_locher))
print("Bayes-Wahrscheinlichkeit Fläche Tacker: " + str(flaeche_bayes_tacker))
print("-----------------------------------------------------------------------------------------")

kanten_bayes_locher, kanten_bayes_tacker = bayes_merkmale(kanten_locher_kat, kanten_tacker_kat, kanten_testbild_kat)
print("Bayes-Wahrscheinlichkeit Kanten Locher: " + str(kanten_bayes_locher))
print("Bayes-Wahrscheinlichkeit Kanten Tacker: " + str(kanten_bayes_tacker))
print("-----------------------------------------------------------------------------------------")

