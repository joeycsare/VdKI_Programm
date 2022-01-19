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
        img = cv.imread("bilder/testbilder/tb_42.jpg")  # hier checken ob quadratisch hoch oder querformat (4:3)

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

def entscheidungsbaum(seitenverh_locher, seitenverh_tacker, flaeche_locher, flaeche_tacker, kanten_locher, kanten_tacker, anz_kategorien):
    # Einzelarrays zu einem Gesamtarray zusammenfassen
    locher_gesamt = np.array([np.transpose(seitenverh_locher), np.transpose(flaeche_locher), np.transpose(kanten_locher)])
    tacker_gesamt = np.array([np.transpose(seitenverh_tacker), np.transpose(flaeche_tacker), np.transpose(kanten_tacker)])

    # Gesamtanzahlen definieren
    anz_locher=len(seitenverh_locher)
    anz_tacker=len(seitenverh_tacker)
    anz_gesamt = anz_locher + anz_tacker

    # Gesamtentropie
    entropie_gesamt=entropy([anz_locher/anz_gesamt, anz_tacker/anz_gesamt], base=2)
    gain = np.zeros(3)
    zaehler_locher = np.zeros((3, anz_kategorien))
    zaehler_tacker = np.zeros((3, anz_kategorien))

    # Schleife um Wurzelknoten zu bestimmen
    for h in range(3): # Schleife um Seitenverhältnis,, Kanten und Fläche zu durchlaufen
        gesamtverh = np.zeros(anz_kategorien)
        locherverh = np.zeros(anz_kategorien)
        tackerverh = np.zeros(anz_kategorien)
        einzelentropie = np.zeros(anz_kategorien)
        einzelentropie_gew_summe = 0
        for i in range(anz_kategorien): # Schleife um alle Kategorien zu durchlaufen und deren Einzelentropie berechnen
            # zaehler_locher = 0
            # zaehler_tacker = 0
            for j in range(anz_locher):
                if locher_gesamt[h, j]==i:
                    zaehler_locher[h, i] = zaehler_locher[h, i]+1
            for j in range(anz_tacker):
                if tacker_gesamt[h, j]==i:
                    zaehler_tacker[h, i] = zaehler_tacker[h, i]+1
            if zaehler_locher[h, i] + zaehler_tacker[h, i] == 0:
                gesamtverh[i] = 0
                locherverh[i] = 0
                tackerverh[i] = 0
                einzelentropie[i] = 0
            else:
                gesamtverh[i] = (zaehler_locher[h, i]+zaehler_tacker[h, i])/anz_gesamt
                locherverh[i] = zaehler_locher[h, i]/ (zaehler_locher[h, i]+zaehler_tacker[h, i])
                tackerverh[i] = zaehler_tacker[h, i]/ (zaehler_locher[h, i]+zaehler_tacker[h, i])
                einzelentropie[i] = entropy([locherverh[i], tackerverh[i]], base=2)
            einzelentropie_gew_summe = einzelentropie_gew_summe + einzelentropie[i]*gesamtverh[i]
            # print(einzelentropie_gew_summe)
        # Gain für die jeweiligen Merkmale (Seitenverh., Fläche und Kante) berechnen
        gain[h] = entropie_gesamt - einzelentropie_gew_summe
        #  print(gesamtverh)
        # print(locherverh)
        # print(tackerverh)
        # print(einzelentropie)

    # print(gain)
    # maximalen Gain herauslesen
    max_gain = max(gain)
    pos_max_gain_array = [p for p, q in enumerate(gain) if q == max_gain]
    pos_max_gain = pos_max_gain_array[0]
    print("Gain 1. Ebene: ", pos_max_gain)

    gain_2 = np.zeros((anz_kategorien, 3))
    zaehler_locher_2_array = np.zeros((anz_kategorien, 3, anz_kategorien))
    zaehler_tacker_2_array = np. zeros((anz_kategorien, 3, anz_kategorien))
    # Schleife um 2. Knoten zu bestimmen
    for g in range(anz_kategorien):
        anz_locher_2 = zaehler_locher[pos_max_gain, g]
        anz_tacker_2 = zaehler_tacker[pos_max_gain, g]
        anz_gesamt_2 = anz_locher_2 + anz_tacker_2
        if anz_gesamt_2 ==0:
            entropie_gesamt_2 = 0
        else:
            entropie_gesamt_2 = entropy([anz_locher_2/anz_gesamt_2, anz_tacker_2/anz_gesamt_2], base=2)

            for h in range(3): # Schleife um Seitenverhältnis,, Kanten und Fläche zu durchlaufen
                zaehler_check = 0
                gesamtverh_summe = 0
                if h != pos_max_gain:
                    gesamtverh = np.zeros(anz_kategorien)
                    locherverh = np.zeros(anz_kategorien)
                    tackerverh = np.zeros(anz_kategorien)
                    einzelentropie = np.zeros(anz_kategorien)
                    einzelentropie_gew_summe = 0
                    for i in range(anz_kategorien): # Schleife um alle Kategorien zu durchlaufen und deren Einzelentropie berechnen
                        zaehler_locher_2 = 0
                        zaehler_tacker_2 = 0
                        for j in range(anz_locher):
                            if locher_gesamt[pos_max_gain, j] == g and locher_gesamt[h, j] == i:
                                zaehler_locher_2 = zaehler_locher_2 + 1
                                zaehler_locher_2_array[g, h, i] = zaehler_locher_2_array[g, h, i] + 1
                        for j in range(anz_tacker):
                            if tacker_gesamt[pos_max_gain, j] == g and tacker_gesamt[h, j] == i:
                                zaehler_tacker_2 = zaehler_tacker_2 + 1
                                zaehler_tacker_2_array[g, h, i] = zaehler_tacker_2_array[g, h, i] + 1
                        if zaehler_locher_2 + zaehler_tacker_2 == 0:
                            gesamtverh[i] = 0
                            locherverh[i] = 0
                            tackerverh[i] = 0
                            einzelentropie[i] = 0
                        else:
                            gesamtverh[i] = (zaehler_locher_2 + zaehler_tacker_2) / anz_gesamt_2
                            locherverh[i] = zaehler_locher_2 / (zaehler_locher_2 + zaehler_tacker_2)
                            tackerverh[i] = zaehler_tacker_2 / (zaehler_locher_2 + zaehler_tacker_2)
                            einzelentropie[i] = entropy([locherverh[i], tackerverh[i]], base=2)
                        einzelentropie_gew_summe = einzelentropie_gew_summe + einzelentropie[i]*gesamtverh[i]
                        gesamtverh_summe = gesamtverh_summe + gesamtverh[i]
                        zaehler_check = zaehler_check + zaehler_tacker_2 + zaehler_locher_2
                        # print(einzelentropie_gew_summe)
                        # Gain für die jeweiligen Merkmale (Seitenverh., Fläche und Kante) berechnen
                        gain_2[g, h] = entropie_gesamt_2 - einzelentropie_gew_summe
                if h == pos_max_gain: # Zähler für Locher und Tacker muss auch beschrieben werden für den Fall, dass Laufvariable h gleich der Kategorie mit dem größten Gain ist
                    for i in range(anz_kategorien): # Schleife um alle Kategorien zu durchlaufen und deren Einzelentropie berechnen
                        for j in range(anz_locher):
                            if locher_gesamt[pos_max_gain, j] == g and locher_gesamt[h, j] == i:
                                zaehler_locher_2_array[g, h, i] = zaehler_locher_2_array[g, h, i] + 1
                        for j in range(anz_tacker):
                            if tacker_gesamt[pos_max_gain, j] == g and tacker_gesamt[h, j] == i:
                                zaehler_tacker_2_array[g, h, i] = zaehler_tacker_2_array[g, h, i] + 1

                # print(gesamtverh_summe)
                # print(zaehler_check)
        # print(entropie_gesamt_2)
        # print(anz_gesamt_2)
    # print(gain_2)
    # print("Anzahl Tacker 2. Ebene: ", zaehler_tacker[pos_max_gain])
    # print("Anzahl Locher 2. Ebene: ", zaehler_locher[pos_max_gain])

    #maximalen Gain für Stufe 2 herauslesen
    pos_max_gain_2 = np.zeros(anz_kategorien, dtype=int)
    for i in range(anz_kategorien):
        laufvariable = gain_2[i]
        max_gain_2_helf = max(laufvariable)
        pos_max_gain_2_array = [p for p, q in enumerate(laufvariable) if q == max_gain_2_helf]
        pos_max_gain_2[i] = pos_max_gain_2_array[0]
    print("Gain 2. Ebene:", pos_max_gain_2)

    # print(zaehler_locher_2_array)
    # print(zaehler_tacker_2_array)

    gain_3 = np.zeros((anz_kategorien, anz_kategorien, 3))
    zaehler_locher_3_array = np.zeros((anz_kategorien, anz_kategorien, 3, anz_kategorien))
    zaehler_tacker_3_array = np.zeros((anz_kategorien, anz_kategorien, 3, anz_kategorien))

    for f in range(anz_kategorien):
        pos_max_gain_2_helf = int(pos_max_gain_2[f])
        for g in range(anz_kategorien):
            anz_locher_3 = zaehler_locher_2_array[f, pos_max_gain_2_helf, g]
            anz_tacker_3 = zaehler_tacker_2_array[f, pos_max_gain_2_helf, g]
            # print("Anz. Locher", anz_locher_3)
            # print("Anz. Tacker", anz_tacker_3)
            anz_gesamt_3 = anz_locher_3 + anz_tacker_3
            if anz_gesamt_3 == 0:
                entropie_gesamt_3 = 0
            else:
                entropie_gesamt_3 = entropy([anz_locher_3 / anz_gesamt_3, anz_tacker_3 / anz_gesamt_3], base=2)

                for h in range(3):  # Schleife um Seitenverhältnis,, Kanten und Fläche zu durchlaufen
                    zaehler_check = 0
                    gesamtverh_summe = 0
                    if h != pos_max_gain and h != pos_max_gain_2_helf:
                        gesamtverh = np.zeros(anz_kategorien)
                        locherverh = np.zeros(anz_kategorien)
                        tackerverh = np.zeros(anz_kategorien)
                        einzelentropie = np.zeros(anz_kategorien)
                        einzelentropie_gew_summe = 0
                        for i in range(anz_kategorien):  # Schleife um alle Kategorien zu durchlaufen und deren Einzelentropie berechnen
                            zaehler_locher_3 = 0
                            zaehler_tacker_3 = 0
                            for j in range(anz_locher):
                                if locher_gesamt[pos_max_gain, j] == f and locher_gesamt[pos_max_gain_2_helf, j] == g and locher_gesamt[h, j] == i:
                                    zaehler_locher_3 = zaehler_locher_3 + 1
                                    zaehler_locher_3_array[f, g, h, i] = zaehler_locher_3_array[f, g, h, i] + 1
                            for j in range(anz_tacker):
                                if tacker_gesamt[pos_max_gain, j] == f and tacker_gesamt[pos_max_gain_2_helf, j] == g and tacker_gesamt[h, j] == i:
                                    zaehler_tacker_3 = zaehler_tacker_3 + 1
                                    zaehler_tacker_3_array[f, g, h, i] = zaehler_tacker_3_array[f, g, h, i] + 1
                            if zaehler_locher_3 + zaehler_tacker_3 == 0:
                                gesamtverh[i] = 0
                                locherverh[i] = 0
                                tackerverh[i] = 0
                                einzelentropie[i] = 0
                            else:
                                gesamtverh[i] = (zaehler_locher_3 + zaehler_tacker_3) / anz_gesamt_3
                                locherverh[i] = zaehler_locher_3 / (zaehler_locher_3 + zaehler_tacker_3)
                                tackerverh[i] = zaehler_tacker_3 / (zaehler_locher_3 + zaehler_tacker_3)
                                einzelentropie[i] = entropy([locherverh[i], tackerverh[i]], base=2)
                            einzelentropie_gew_summe = einzelentropie_gew_summe + einzelentropie[i] * gesamtverh[i]
                            gesamtverh_summe = gesamtverh_summe + gesamtverh[i]
                            zaehler_check = zaehler_check + zaehler_tacker_2 + zaehler_locher_2
                            # print(einzelentropie_gew_summe)
                            # Gain für die jeweiligen Merkmale (Seitenverh., Fläche und Kante) berechnen
                            gain_3[f, g, h] = entropie_gesamt_3 - einzelentropie_gew_summe
                    else:  # Zähler für Locher und Tacker muss auch beschrieben werden für den Fall, dass Laufvariable h gleich der Kategorie mit dem größten Gain ist
                        for i in range(anz_kategorien):  # Schleife um alle Kategorien zu durchlaufen und deren Einzelentropie berechnen
                            for j in range(anz_locher):
                                if locher_gesamt[pos_max_gain, j] == f and locher_gesamt[pos_max_gain_2_helf, j] == g and locher_gesamt[h, j] == i:
                                    zaehler_locher_3_array[f, g, h, i] = zaehler_locher_3_array[f, g, h, i] + 1
                            for j in range(anz_tacker):
                                if tacker_gesamt[pos_max_gain, j] == f and tacker_gesamt[pos_max_gain_2_helf, j] == g and tacker_gesamt[h, j] == i:
                                    zaehler_tacker_3_array[f, g, h, i] = zaehler_tacker_3_array[f, g, h, i] + 1

    # print(zaehler_locher_3_array)
    # print(zaehler_tacker_3_array)
    # print(gain_3)
    pos_max_gain_3 = np.zeros((anz_kategorien, anz_kategorien), dtype=int)
    for i in range(anz_kategorien):
        for j in range(anz_kategorien):
            laufvariable = gain_3[j, i]
            max_gain_3_helf = max(laufvariable)
            pos_max_gain_3_array = [p for p, q in enumerate(laufvariable) if q == max_gain_3_helf]
            pos_max_gain_3[j, i] = pos_max_gain_3_array[0]
    # print(pos_max_gain_3)

    # Alternative Berechnung Position max. Gain Ebene 3
    pos_max_gain_3_alternativ = np.zeros((anz_kategorien, anz_kategorien), dtype=int)
    for i in range(anz_kategorien):
        for j in range(anz_kategorien):
            if pos_max_gain != 0 and int(pos_max_gain_2[i]) != 0:
                pos_max_gain_3_alternativ[i, j] = 0
            if pos_max_gain != 1 and int(pos_max_gain_2[i]) != 1:
                pos_max_gain_3_alternativ[i, j] = 1
            if pos_max_gain != 2 and int(pos_max_gain_2[i]) != 2:
                pos_max_gain_3_alternativ[i, j] = 2
    print("Gain 3. Ebene:\n", pos_max_gain_3_alternativ)

    # Zaehler Locher Tacker für Export vorbereiten: Ebene (Zähler) 1
    anz_locher_1_out = zaehler_locher[pos_max_gain]
    print("Locher Out 1. Ebene:", anz_locher_1_out)
    anz_tacker_1_out = zaehler_tacker[pos_max_gain]
    print("Tacker Out 1. Ebene:", anz_tacker_1_out)

    # Zähler Locher Tacker für Export vorbereiten: Ebene (Zähler) 2
    anz_locher_2_out = np.zeros((anz_kategorien, anz_kategorien))
    anz_tacker_2_out = np.zeros((anz_kategorien, anz_kategorien))
    for i in range(anz_kategorien):
        anz_locher_2_out[i] = zaehler_locher_2_array[i, int(pos_max_gain_2[i])]
        anz_tacker_2_out[i] = zaehler_tacker_2_array[i, int(pos_max_gain_2[i])]
    print("Locher Out 2. Ebene:\n", anz_locher_2_out)
    print("Tacker Out 2. Ebene:\n", anz_tacker_2_out)

    # Zähler Locher Tacker für Export vorbereiten: Ebene (Zähler) 3
    anz_locher_3_out = np.zeros((anz_kategorien, anz_kategorien, anz_kategorien))
    anz_tacker_3_out = np.zeros((anz_kategorien, anz_kategorien, anz_kategorien))
    for i in range(anz_kategorien):
        for j in range(anz_kategorien):
            anz_locher_3_out[i, j] = zaehler_locher_3_array[i, j, int(pos_max_gain_3_alternativ[i, j])]
            anz_tacker_3_out[i, j] = zaehler_tacker_3_array[i, j, int(pos_max_gain_3_alternativ[i, j])]
    # print("Locher Out 3. Ebene:\n", anz_locher_3_out)
    # print("Tacker Out 3. Ebene:\n", anz_tacker_3_out)

    return(anz_locher_1_out, anz_locher_2_out, anz_locher_3_out, anz_tacker_1_out, anz_tacker_2_out, anz_tacker_3_out, pos_max_gain, pos_max_gain_2, pos_max_gain_3_alternativ)

def entscheidungsbaum_testbild(anz_locher_1_out, anz_locher_2_out, anz_locher_3_out, anz_tacker_1_out, anz_tacker_2_out, anz_tacker_3_out, gain, gain_2, gain_3, seitenverh_testbild_kat, flaeche_testbild_kat, kanten_testbild_kat):
    testbild = np.zeros(3, dtype=int)
    testbild[0] = int(seitenverh_testbild_kat[0])
    testbild[1]= int(flaeche_testbild_kat[0])
    testbild[2] = int(kanten_testbild_kat[0])
    print("Seitenverhältnis Testbild Kat.:", testbild[0])
    print("Fläche Testbild Kat.:", testbild[1])
    print("Kanten Testbild Kat.:", testbild[2])

    kat_1 = testbild[gain]
    kat_2 = testbild[gain_2[kat_1]]
    kat_3 = testbild[gain_3[kat_1, kat_2]]
    print(kat_1)
    print(kat_2)
    print(kat_3)

    # print(anz_locher_3_out)
    # print(anz_tacker_3_out)

    #Wahrscheinlichkeit Berechnung: wenn Locher = 1,0 wenn Tacker = 0,0
    if anz_locher_1_out[kat_1] == 0 and anz_tacker_1_out[kat_1] == 0:
        print("Error: keine Testbilder in dieser Kategorie")
    elif anz_locher_1_out[kat_1] == 0 or anz_tacker_1_out[kat_1] == 0:
        prop_locher = anz_locher_1_out[kat_1] / (anz_locher_1_out[kat_1] + anz_tacker_1_out[kat_1])
    else:
        if anz_locher_2_out[kat_1, kat_2] == 0 and anz_tacker_2_out[kat_1, kat_2] == 0:
            prop_locher = anz_locher_1_out[kat_1] / (anz_locher_1_out[kat_1] + anz_tacker_1_out[kat_1])
        elif anz_locher_2_out[kat_1, kat_2] == 0 or anz_tacker_2_out[kat_1, kat_2] == 0:
            prop_locher = anz_locher_2_out[kat_1, kat_2] / (anz_locher_2_out[kat_1, kat_2] + anz_tacker_2_out[kat_1, kat_2])
        else:
            if anz_locher_3_out[kat_1, kat_2, kat_3] == 0 and anz_tacker_3_out[kat_1, kat_2, kat_3] == 0:
                prop_locher = anz_locher_2_out[kat_1, kat_2] / (anz_locher_2_out[kat_1, kat_2] + anz_tacker_2_out[kat_1, kat_2])
            else:
                prop_locher = anz_locher_3_out[kat_1, kat_2, kat_3] / (anz_locher_3_out[kat_1, kat_2, kat_3] + anz_tacker_3_out[kat_1, kat_2, kat_3])

    if prop_locher >0.45 and prop_locher < 0.55:
        print("Weder Locher noch Tacker erkannt. Locher-Wahrscheinlichkeit:" , prop_locher)
    else:
        print("Wahrscheinlichkeit, dass Testbild Locher ist:", prop_locher)
        print("Wahrscheinlichkeit, dass Testbild Tacker ist:", 1 - prop_locher)


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
Hauptprogramm: Entscheidungsbaum
====================================================================================================================
"""
anz_locher_1_out, anz_locher_2_out, anz_locher_3_out, anz_tacker_1_out, anz_tacker_2_out, anz_tacker_3_out, gain, gain_2, gain_3 = entscheidungsbaum(seitenverh_locher_kat, seitenverh_tacker_kat, flaeche_locher_kat, flaeche_tacker_kat, kanten_locher_kat, kanten_tacker_kat, anz_kategorien)

entscheidungsbaum_testbild(anz_locher_1_out, anz_locher_2_out, anz_locher_3_out, anz_tacker_1_out, anz_tacker_2_out, anz_tacker_3_out, gain, gain_2, gain_3, seitenverh_testbild_kat, flaeche_testbild_kat, kanten_testbild_kat)