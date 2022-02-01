import os
import random
import csv
import cv2 as cv
import numpy as np
import math


class ImageLoader():
    __directoryNames = []
    __allImageNames = []
    __allImagePaths = []
    __allDirPaths = []
    __Data = []
    __sampleEntrys = []
    __sampleEntrysPath = []

    def __init__(self, firstDirectory=str, useAbsolutePath=bool):
        if useAbsolutePath == None:
            self.AddDirRealPath(firstDirectory)
        elif useAbsolutePath == False:
            self.AddDirRealPath(firstDirectory)
        else:
            self.AddDirAbsPath(firstDirectory)
        self.MakeSampleEntrys(5)

    def AddDirAbsPath(self, directory):
        self.__directoryNames.append(directory)
        self.__MakeDictionary()
        print(f'Added Directory {directory}')

    def AddDirRealPath(self, directory):
        self.__directoryNames.append(os.getcwd() + '\\' + directory)
        self.__MakeDictionary()
        print(f'Added Directory {directory}')

    def DeleteDir(self, directory):
        self.__directoryNames.remove(directory)
        self.__MakeDictionary()

    def __MakeDictionary(self):
        allList = []
        pathList = []
        nameList = []
        dirList = []
        a = 0
        for dir in self.__directoryNames:
            files = os.listdir(dir)
            for file in files:
                if os.path.isfile(os.path.join(dir, file)):
                    allList.append({'index': a, 'name': os.path.basename(file), 'type': os.path.basename(
                        dir), 'assignment': os.path.dirname(dir), 'directoryPath': dir, 'path': os.path.join(dir, file)})
                    nameList.append(os.path.basename(file))
                    pathList.append(os.path.join(dir, file))
                    dirList.append(os.path.basename(dir))
                    a = a + 1
        self.__allImageNames = nameList
        self.__allImagePaths = pathList
        self.__allDirPaths = dirList
        self.__Data = allList

    def MakeSampleEntrys(self, testsize=1):
        if testsize > 0:
            testList = []
            patherList = []
            holdList = self.__Data
            if testsize > len(holdList):
                return holdList
            while (len(testList) < testsize):
                a = random.randint(0, len(holdList)-1)
                entry = holdList[a]
                testList.append(entry)
                patherList.append(entry.get('path'))
            self.__sampleEntrys = testList
            self.__sampleEntrysPath = patherList

    def PrintSampleEntrys(self):
        print('---------------------------------------------------------------------------------------------------')
        for entry in self.__sampleEntrys:
            for key in entry:
                print(key, str(entry.get(key)))
            print('------------------------------------------------------------')
        print('---------------------------------------------------------------------------------------------------')

    def AllData(self):
        return self.__Data

    def AddColumns(self, Column=[]):
        j = 0
        for entry in self.__Data:
            entry.update(Column[j])
            entry = dict(sorted(entry.items(), key=lambda kv: kv[0]))
            j += 1

    def DeleteColumn(self, Key):
        for entry in self.__Data:
            try:
                entry.pop(Key)
            except:
                print(Key, " is not a key in this Column")
            entry = dict(sorted(entry.items(), key=lambda kv: kv[0]))

    def AddRow(self, Row={}):
        self.__Data.append(Row)

    def DeleteRow(self, RowNumber=0):
        self.__Data.pop(RowNumber)

    def PrintAllToCSV(self, fileName = 'allCSV'):
        with open(fileName, 'w', encoding='UTF8', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            keyList = sorted(self.__Data[0].keys())
            csvwriter.writerow(keyList)
            for entry in self.__Data:
                line = []
                for key in keyList:
                    line.append(str(entry.get(key)))
                csvwriter.writerow(line)
        print('-------------All CSV Done------------')

    def AddColumnsToSample(self, Column=[]):
        j = 0
        for entry in self.__sampleEntrys:
            if j >= len(Column):
                j -= 1
            entry.update(Column[j])
            entry = dict(sorted(entry.items(), key=lambda kv: kv[0]))
            j += 1

    def DeleteColumnFromSample(self, Key):
        for entry in self.__sampleEntrys:
            try:
                entry.pop(Key)
            except:
                print(Key, " is not a key in this Column")
            entry = dict(sorted(entry.items(), key=lambda kv: kv[0]))

    def AddRowToSample(self, Row={}):
        self.__sampleEntrys.append(Row)

    def DeleteRowFromSample(self, RowNumber=0):
        self.__sampleEntrys.pop(RowNumber)

    def PrintSamplesToCSV(self, fileName = 'sampleCSV'):
        with open(fileName, 'w', encoding='UTF8', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            keyList = sorted(self.__sampleEntrys[0].keys())
            csvwriter.writerow(keyList)
            for entry in self.__sampleEntrys:
                line = []
                for key in keyList:
                    line.append(str(entry.get(key)))
                csvwriter.writerow(line)
        print('-------------Sample CSV Done------------')

    @property
    def sampleEntrys(self):
        return self.__sampleEntrys

    @property
    def sampleEntrysPath(self):
        return self.__sampleEntrysPath

    @property
    def dirNames(self):
        return self.__allDirPaths

    @property
    def allImgNames(self):
        return self.__allImageNames

    @property
    def allImgPaths(self):
        return self.__allImagePaths


class ImageChanger():

    __sampleImageList = []
    __sampleTypeList = []
    __originalList = []
    __lastList = []
    __contureValues = []
    __colorList = []
    __errorList = []

    def __init__(self, sampleList=[], holdtime=20, wait=False) -> None:
        if len(sampleList) > 0:
            self.__sampleImageList = []
            self.__originalList = []
            self.__lastList = []
            e = 0
            for sam in sampleList:
                try:
                    self.__sampleImageList.append(cv.imread(sam))
                    self.__originalList.append(cv.imread(sam))
                    self.__lastList.append(cv.imread(sam))
                    self.__sampleTypeList.append(os.path.dirname(sam))
                except Exception as f:
                    self.__errorList.append(
                        'ERROR: Readin ' + str(e) + '---------------')
                    print(repr(f))
            self.holdtime = holdtime
            if wait:
                cv.imshow('construct prewiev', self.__sampleImageList[0])
                cv.waitKey(100 * self.holdtime)

    def Revert(self):
        self.__sampleImageList = self.__lastList

    def Hold(self):
        cv.waitKey(0)

    def Reset(self):
        self.__sampleImageList = self.__originalList

    def ShowRandomImage(self):
        a = random.randint(0, len(self.__sampleImageList)-1)
        cv.imshow('Random Image', self.__sampleImageList[a])
        cv.waitKey(100 * self.holdtime)

    def GetRandomImage(self):
        a = random.randint(0, len(self.__sampleImageList)-1)
        return self.__sampleImageList[a]

    def ShowAll(self):
        c = 0
        s = ''
        for frame in self.__sampleImageList:
            c = c + 1
            s = 'Of all Images: ' + str(c)
            cv.imshow(s, frame)
        cv.waitKey(100 * self.holdtime)

    def ShowIndexImage(self, index=0):
        cv.imshow('Index Image', self.__sampleImageList[index])
        cv.waitKey(100 * self.holdtime)

    def ShowImage(self, frame, kk=0):
        cv.imshow('Frame Image', frame)
        cv.waitKey(kk * 100)

    def ShowImageLoop(self):
        a = 0
        while a < len(self.__sampleImageList):
            cv.imshow('Index Image loop ' + str(a), self.__sampleImageList[a])
            a = a+1
            cv.waitKey(100 * self.holdtime)

    def FrameByScale(self, scale=0, interpol=cv.INTER_AREA):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            try:
                intwidth = int(frame.shape[1]*scale)
                intheight = int(frame.shape[0]*scale)
                dimensions = (intwidth, intheight)
                e = e + 1
                img = cv.resize(frame, dimensions, interpolation=interpol)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: FrameByScale ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList
        self.__colorList = retList

    def FrameByWidth(self, targetwidth=500, interpol=cv.INTER_AREA):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            try:
                useScale = frame.shape[0]/frame.shape[1]
                intwidth = int(targetwidth)
                intheight = int(targetwidth * useScale)
                dimensions = (intwidth, intheight)
                e = e + 1
                img = cv.resize(frame, dimensions, interpolation=interpol)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: FrameByWidth ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList
        self.__colorList = retList

    def FrameByHeight(self, targetheight=500, interpol=cv.INTER_AREA):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            try:
                useScale = frame.shape[1]/frame.shape[0]
                intheight = int(targetheight)
                intwidth = int(targetheight * useScale)
                dimensions = (intwidth, intheight)
                e = e + 1
                img = cv.resize(frame, dimensions, interpolation=interpol)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: FrameByHeight ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList
        self.__colorList = retList

    def FrameByValues(self, targetwidth, targetheight, interpol=cv.INTER_AREA):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = cv.resize(
                    frame, (targetwidth, targetheight), interpolation=interpol)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: FrameByValues ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList
        self.__colorList = retList

    def Blur(self, grainsize=3):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = cv.GaussianBlur(
                    frame, (grainsize, grainsize), cv.BORDER_DEFAULT)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Blur ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def Dilation(self, kernelsize=5, iterate=1):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                kernel = cv.getStructuringElement(
                    cv.MORPH_ELLIPSE, (kernelsize, kernelsize))
                img = cv.dilate(frame, kernel,
                                iterations=iterate)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Dilate ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def Cropping(self, y1=50, x1=50, y2=450, x2=450):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = frame[x1:y1, x2:y2]
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Cropping ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def Bordering(self, bordersize=10, color=0):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                if frame[5][5] > 127:
                    color = 255
                else:
                    color = 0
                img = cv.copyMakeBorder(frame, bordersize, bordersize, bordersize, bordersize, cv.BORDER_CONSTANT, value=color)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Bordering ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList
        retList = []
        e = 0
        for frame in self.__colorList:
            e = e + 1
            try:
                img = cv.copyMakeBorder(
                    frame, bordersize, bordersize, bordersize, bordersize, cv.BORDER_CONSTANT, value=250)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Bordering color ' + str(e) + '--------------- ' + repr(f))
        self.__colorList = retList

    def InvertCheck(self):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                m_b = np.average(frame)
                if m_b > 127:
                    _, frame = cv.threshold(frame, 127, 255, cv.THRESH_BINARY_INV)
                retList.append(frame)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: invert ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def Erode(self, kernelsize=3, iterate=1):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                kernel = cv.getStructuringElement(
                    cv.MORPH_ELLIPSE, (kernelsize, kernelsize))
                img = cv.erode(frame, kernel,
                               iterations=iterate)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Erode ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def Gray(self):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Gray ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def HSV(self):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: HSV ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def ColorspaceCorrection(self, transformSpace=cv.COLOR_BGR2LAB, undoSpace=cv.COLOR_LAB2BGR,  Limit=3.0, GridSize=8, applyTo=0, showPrewiev=False):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                lab = cv.cvtColor(frame, transformSpace)
                a, b, c = cv.split(lab)
                liste = [a, b, c]
                clahe = cv.createCLAHE(
                    clipLimit=Limit, tileGridSize=(GridSize, GridSize))
                liste[applyTo] = clahe.apply(liste[applyTo])
                limg = cv.merge(liste)
                final = cv.cvtColor(limg, undoSpace)
                if e == 1 & showPrewiev:
                    cv.imshow('Previous', frame)
                    cv.imshow('New ColorSpace', lab)
                    cv.imshow('A_channel', a)
                    cv.imshow('B_channel', b)
                    cv.imshow('C_channel', c)
                    cv.imshow('limg', limg)
                    cv.imshow('CLAHE output', liste[applyTo])
                    cv.imshow('New', final)
                    cv.waitKey(300 * self.holdtime)
                retList.append(final)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: ColorspaceCorrection ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def ColorspaceToColor(self, targetColor=0, showPrewiev=False):
        retList = []
        e = 0
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                lab = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                a, b, c = cv.split(lab)
                anew = np.zeros(
                    (frame.shape[0], frame.shape[1], 1), dtype=np.uint8)+targetColor
                limg = cv.merge((anew, b, c))
                final = cv.cvtColor(limg, cv.COLOR_HSV2BGR)
                if e == 1 & showPrewiev:
                    cv.imshow('Previous', frame)
                    cv.imshow('New ColorSpace', lab)
                    cv.imshow('A_channel', a)
                    cv.imshow('B_channel', b)
                    cv.imshow('C_channel', c)
                    cv.imshow('ohne Hue', limg)
                    cv.imshow('New', final)
                    cv.waitKey(300 * self.holdtime)
                retList.append(final)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: ColorspaceCorrection ' + str(e) + '--------------- ' + repr(f))

    def HLS(self):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = cv.cvtColor(frame, cv.COLOR_BGR2HLS)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: HLS ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def LAB(self):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: LAB ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def Normieren(self):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Normiert ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def BilateralFilter(self, sigCol=100, sigSpc=100):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = cv.bilateralFilter(frame, 9, sigCol, sigSpc)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Normiert ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def CannyNorm(self, edge=3, tres1=125, tres2=175):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = cv.Canny(frame, tres1, tres2,
                               edges=edge, L2gradient=True)
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Canny ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def CannyTest(self, tres1=0, tres2=0):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        try:
            cv.namedWindow('Threshold Tracking')
            cv.createTrackbar(
                'threshold_1', 'Threshold Tracking', 0, 255, self.__nothing)
            cv.createTrackbar(
                'threshold_2', 'Threshold Tracking', 0, 255, self.__nothing)
            while True:
                tres1 = cv.getTrackbarPos('threshold_1', 'Threshold Tracking')
                tres2 = cv.getTrackbarPos('threshold_2', 'Threshold Tracking')
                img = cv.Canny(
                    self.__sampleImageList[0], tres1, tres2, L2gradient=True)
                cv.imshow('Threshold Preview', img)
                waitKEY = cv.waitKey(1)
                if waitKEY == 27:
                    print('new Values -------------------------------------- ')
                    print('threshold1 -- ', tres1)
                    print('threshold2 -- ', tres2)
                    print('------------------------------------------------- ')
                    break
        except Exception as f:
            self.__errorList.append(
                'ERROR: CannyPreview ' + str(e) + '--------------- ' + repr(f))

    def AdaptiveCanny(self, sigma=0.8, contLenght=10, edge=3):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                contas = []
                wide = 0
                while len(contas) > contLenght:
                    # lower threshold (Grenze)
                    lower = wide + int(max(50, (1.0 - sigma) * 170))
                    # upper threshold (Grenze)
                    upper = wide + int(min(255, (1.0 + sigma) * 127))
                    # OpenCV, die de Kante sucht
                    img = cv.Canny(frame, lower, upper,
                                   edges=edge, L2gradient=True)
                    contas, hier = cv.findContours(
                        img, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
                    wide += 5
                retList.append(img)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Canny ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def Threshold(self, track=False, tresh=127, type1=cv.THRESH_BINARY, type2=cv.THRESH_OTSU, invert=True):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                T1 = cv.threshold(frame, tresh, 255, type1, type2)
                if invert:
                    T1 = cv.threshold(frame, T1[0], 255, cv.THRESH_BINARY_INV)
                retList.append(T1[1])
            except Exception as f:
                self.__errorList.append(
                    'ERROR: ThresholdDual ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def ThresholdTest(self, tresh=0, type1=cv.THRESH_BINARY, type2=cv.THRESH_OTSU, invert=True):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        try:
            cv.namedWindow('Threshold Tracking')
            cv.createTrackbar(
                'threshold', 'Threshold Tracking', 0, 255, self.__nothing)
            while True:
                tresh = cv.getTrackbarPos(
                    'threshold', 'Threshold Tracking')
                T1 = cv.threshold(
                    self.__sampleImageList[0], tresh, 255, type1, type2)
                if invert:
                    T1 = cv.threshold(
                        self.__sampleImageList[0], T1[0], 255, cv.THRESH_BINARY_INV)
                cv.imshow('Threshold Preview -- close with x', T1[1])
                waitKEY = cv.waitKey(1)
                if waitKEY == 27:
                    print('new Values -------------------------------------- ')
                    print('threshold -- ', tresh)
                    print('------------------------------------------------- ')
                    break
        except Exception as f:
            self.__errorList.append(
                'ERROR: ThresholdTest ' + str(e) + '--------------- ' + repr(f))

    def ThresholdDualAdapt(self, contLenght=5, type1=cv.THRESH_BINARY, type2=cv.THRESH_OTSU, invert=True):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                contas = []
                wide = 0
                while (len(contas) < contLenght):  # & (wide < 255):
                    img = cv.threshold(frame, wide, 255, type1, type2)
                    contas, hier = cv.findContours(
                        img[1], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
                    cv.imshow('test', img[1])
                    cv.waitKey(1000)
                    wide += 5
                retList.append(frame)
            except Exception as f:
                self.__errorList.append(
                    'ERROR: Threshold adapt ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def ColorAdapt(self, showPrewiev=False, sigma=0.9, type1=cv.THRESH_BINARY, type2=cv.THRESH_OTSU):
        retList = []
        e = 0
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                hue, sat, bright = cv.split(hsv)
                # farbe ist nur da wo object ist, senkt aber brightness
                # farbe normalisieren für durchschnitt
                # neue farbe mit threshold absolut machen

                holz = np.zeros(
                    (hsv.shape[0], hsv.shape[1], 1), dtype=np.uint8)
                rows, cols = hue.shape  # holziness
                for i in range(rows):
                    for j in range(cols):
                        h, s, v = hsv[i, j]
                        if h < 70 | h > 220 & s > 50:
                            holz[i, j] = 250

                bright = cv.bitwise_not(bright)  # brightness invertieren
                # brightness auf sättigung addieren
                bright2 = cv.add(bright, sat)
                # threshold anpassen für beides

                m = np.average(bright2)  # Medianwert
                lower = sigma * m
                T1 = cv.threshold(bright2, lower, 255, type1, type2)

                if e == 1 & showPrewiev:
                    cv.imshow('Previous', frame)
                    cv.imshow('Sattigung', sat)
                    cv.imshow('Brightness', bright)
                    cv.imshow('Brightness2', bright2)
                    cv.imshow('T1', T1[1])
                    cv.imshow('Holz', holz)
                    cv.waitKey(300 * self.holdtime)
                retList.append(bright2)

            except Exception as f:
                self.__errorList.append(
                    'ERROR: ColorspaceCorrection ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def BrightTresh(self, showPrewiev=False, sigma=0.9, type1=cv.THRESH_BINARY, type2=cv.THRESH_OTSU):
        retList = []
        e = 0
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                m = np.average(frame)  # Mittelwert
                border = 127 - (sigma * (127 - m))
                T1 = cv.threshold(frame, border, 255, type1, type2)
                retList.append(T1[1])
            except Exception as f:
                self.__errorList.append(
                    'ERROR: BrightTresh ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def __nothing(self, x):
        pass

    def __sortFunc(self, x):
        a = cv.contourArea(cv.convexHull(x))
        return a

    def ConturMerkmale(self, type1=cv.RETR_EXTERNAL, type2=cv.CHAIN_APPROX_NONE, epsyValue=0.01, printing=False):
        retList = []
        self.__lastList = self.__sampleImageList
        e = 0
        g = 0
        for frame in self.__sampleImageList:  # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
            # if True:
            try:
                e = e + 1
                drawing = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                contours, _ = cv.findContours(frame, type1, type2)

                sorted_contours = sorted(contours, key=self.__sortFunc, reverse=False)

                sortet_convex_contoures = []
                sortet_convex_contoures_noReturn = []
                sortet_approx_contoures = []
                for c in sorted_contours:
                    sortet_convex_contoures.append(cv.convexHull(c))
                    sortet_convex_contoures_noReturn.append(cv.convexHull(c,returnPoints=False))
                    sortet_approx_contoures.append(cv.approxPolyDP(c, epsyValue*cv.arcLength(c, True), True))

                cnt = sorted_contours[len(sorted_contours)-1]
                cnt_c = sortet_convex_contoures[len(sorted_contours)-1]
                cnt_cnR = sortet_convex_contoures_noReturn[len(sorted_contours)-1]
                cnt_a = sortet_approx_contoures[len(sorted_contours)-1]

                cnt2 = sorted_contours[len(sorted_contours)-2]
                cnt2_c = sortet_convex_contoures[len(sorted_contours)-2]
                cnt2_a = sortet_approx_contoures[len(sorted_contours)-2]

            # humoments
                # huMoments = cv.HuMoments(cv.moments(cnt))

                # for i in range(0,7):
                #     huMoments[i] = -1* math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))

                compareImage = np.zeros(frame.shape, dtype=np.uint8)
                compareImage1 = cv.rectangle(compareImage, (150,100), (250,300), 255, -1)

                compareImage = np.zeros(frame.shape, dtype=np.uint8)
                compareImage2 = cv.circle(compareImage, (200,200), 50, 255, -1)
                cntImage1, _ = cv.findContours(compareImage1, type1, type2)
                cntImage2, _ = cv.findContours(compareImage2, type1, type2)

                # cv.imshow('comp1: ', compareImage1)
                # cv.imshow('comp2: ', compareImage2)

                # mask
                mask = np.zeros(frame.shape, np.uint8)
                cv.drawContours(mask, [cnt], 0, 255, -1)

                orb = cv.ORB_create()
                keyPoints, description = orb.detectAndCompute(frame, None)


            # First Conture data ------------------------------------------------------------------------------------------------------------
                area = cv.contourArea(cnt)
                perimeter = cv.arcLength(cnt, True)

                area_c = cv.contourArea(cnt_c)
                perimeter_c = cv.arcLength(cnt_c, True)

                minCircle = cv.minEnclosingCircle(cnt)
                (x,y),radius = minCircle
                center = (int(x),int(y))
                radius = int(radius)

                mainMoment = cv.moments(cnt)
                cx = int(mainMoment['m10']/mainMoment['m00'])
                cy = int(mainMoment['m01']/mainMoment['m00'])

                convexMoment = cv.moments(cnt_c)
                cx_c = int(convexMoment['m10']/convexMoment['m00'])
                cy_c = int(convexMoment['m01']/convexMoment['m00'])

                minRect = cv.minAreaRect(cnt)
                mx = int(minRect[0][0])
                my = int(minRect[0][1])
                mw = int(minRect[1][0])
                mh = int(minRect[1][1])
                mAn = int(minRect[2])

            # First Conture Drawing ----------------
                cv.drawContours(drawing, sorted_contours, len(sorted_contours)-1, (255, 255, 255))  # Contour
                cv.drawContours(drawing, sortet_approx_contoures, len(sorted_contours)-1, (0, 255, 0))  # Approx Contour
                cv.drawContours(drawing, sortet_convex_contoures, len(sorted_contours)-1, (255, 0, 0))  # Convex Contour

                minbox = np.intp(cv.boxPoints(minRect))  # Box(lila)
                
                cv.drawContours(drawing, [minbox], 0, (255, 0, 255))
                cv.circle(drawing,center,radius,(255, 255, 0), 2)
                cv.circle(drawing, (mx, my), 3, (255, 0, 255))# Mittelpunkt(grün) der Box
                cv.circle(drawing, center, 3, (255, 255, 0))# Mittelpunkt (türkis) des minimalen Kreises

                cv.circle(drawing, (cx, cy), 3, (255, 255, 255))# Schwerpunkt(rot) der Fläche der Kontur
                cv.circle(drawing, (cx_c, cy_c), 3, (255, 0, 0)) # Schwerpunkt(gelb) der Fläche der Convexen Kontur
                
            # Second Conture data -----------------------------------------------------------------------------------------------------------
                area2 = cv.contourArea(cnt2)
                perimeter2 = cv.arcLength(cnt2, True)

                area2_c = cv.contourArea(cnt2_c)
                perimeter2_c = cv.arcLength(cnt2_c, True)

                minCircle2 = cv.minEnclosingCircle(cnt)
                (x2,y2),radius2 = minCircle2
                center2 = (int(x2),int(y2))
                radius2 = int(radius2)

                mainMoment2 = cv.moments(cnt2)
                cx2 = int(mainMoment2['m10']/mainMoment2['m00'])
                cy2 = int(mainMoment2['m01']/mainMoment2['m00'])

                mainMoment2 = 0
                cx2 = 0
                cy2 = 0
                
                convexMoment2 = cv.moments(cnt2_c)
                cx2_c = int(convexMoment2['m10']/convexMoment2['m00'])
                cy2_c = int(convexMoment2['m01']/convexMoment2['m00'])
                
                convexMoment2 = 0
                cx2_c = 0
                cy2_c = 0

                minRect2 = cv.minAreaRect(cnt2)
                mx2 = int(minRect2[0][0])
                my2 = int(minRect2[0][1])
                mw2 = int(minRect2[1][0])
                mh2 = int(minRect2[1][1])
                mAn2 = int(minRect2[2])

            # Second Conture Drawing ----------------
                cv.drawContours(drawing, sorted_contours, len(sorted_contours)-2, (255, 255, 255))  # Contour
                cv.drawContours(drawing, sortet_approx_contoures, len(sorted_contours)-2, (0, 255, 0))  # Approx Contour
                cv.drawContours(drawing, sortet_convex_contoures, len(sorted_contours)-2, (255, 0, 0))  # Convex Contour

                minbox2 = np.intp(cv.boxPoints(minRect2))  # Box(lila)
                
                cv.drawContours(drawing, [minbox2], 0, (255, 0, 255))
                cv.circle(drawing,center2,radius2,(255, 255, 0), 2)
                cv.circle(drawing, (mx2, my2), 3, (255, 0, 255))# Mittelpunkt(grün) der Box
                cv.circle(drawing, center2, 3, (255, 255, 0))# Mittelpunkt (türkis) des minimalen Kreises

                cv.circle(drawing, (cx2, cy2), 3, (255, 255, 255))# Schwerpunkt(rot) der Fläche der Kontur
                cv.circle(drawing, (cx2_c, cy2_c), 3, (255, 0, 0)) # Schwerpunkt(gelb) der Fläche der Convexen Kontur
                
            # Merkmalsberechnung-------------------------------------------------------------------------------------------------------------

            # conture 1 --------------
                seradity = float(perimeter_c)/perimeter                                        # wie zerfurcht ist das objekt
                solidity = float(area)/area_c                                                   # wie convex ist das objekt
                
                equi_diameter = np.sqrt(4*area/np.pi)                                           # durchmesser eines kreises mit gleicher fläche

                minimal_ratio = max(minRect[1]) / min(minRect[1])                               # verhältnisse der seitenlängen

                momentpointDistance = cv.pointPolygonTest(cnt, (cx, cy), True)                  # abstand des Schwerpunkts zur nächsten kante
                middlepointDistance = cv.pointPolygonTest(cnt, (mx, my), True)/minimal_ratio    # normierter abstand des Mittelpunkts zur nächsten kante 

                steiner = np.sqrt(((mx-cx)**2)+((my-cy)**2))                                    # abstand des Schwerpunkts zum Mittelpunkt

                approxAnzahl = len(cnt_a)                                                       # Menge der nötigen Approximationssegmente

            # conture 1 und 2 --------------
                seradity2 = float(perimeter2_c)/perimeter2                                      # wie zerfurcht ist das die zweite kontur 
                solidity2 = float(area2)/area2_c                                                # wie convex ist die zweite kontur         
                
                angle = abs(mAn-mAn2)                                                           # Winkel zwischen den Boxen der beiden größten Conturen 
                minimal_ratio2 = max(minRect2[1]) / min(minRect2[1])                            # verhältnisse der seitenlängen

                areaRatio = area2/area                                                          # größenverhältnis der conturen

                contureDistance = (np.sqrt(((cx2-cx)**2)+((cy2-cy)**2))/max(minRect[1]))  # Abstand der Mittelpunkte der beiden größten konturen im verhältnis zu gesamtlänge
            
            # Overall   --------------
                
                conturenAnzahl = len(contours)                                                  # wie viele conturen wurden gefunden
                
                rectangleLike = cv.matchShapes(cnt, cntImage1[0], cv.CONTOURS_MATCH_I2, 0)      # wie rectangular ist das bild
                circleLike = cv.matchShapes(cnt, cntImage2[0], cv.CONTOURS_MATCH_I2, 0)         # wie circular ist das bild

                keyPointsAnzahl = len(keyPoints)                                                #anzahl der gefundenen keypoints


                mean_val,_,_,_ = cv.mean(frame, mask=mask)                                      # mean der Fläche in der kontur
                # colorMap,_,_ = cv.split(cv.cvtColor(self.__colorList[g], cv.COLOR_BGR2HSV))
                # mean_color,_,_,_ = cv.mean(colorMap,mask=mask)

                farPoint = 0                                                                    # größte entfernung von der convexen kontur                                                                         # fehleranfällig
                defects = cv.convexityDefects(cnt, cnt_cnR)
                try:
                    for i in range(defects.shape[0]):
                        _, _, _, d = defects[i, 0]
                        if d > farPoint:
                            farPoint = d
                except Exception as t:
                    self.__errorList.append('ERROR: Farpoint 1 ' + str(e) + '--------------- ' + repr(t))

                objectType = os.path.basename(self.__sampleTypeList[g])

                dic = {
                     'seradity':seradity,
                    'solidity':solidity,
                    'minimal_ratio':minimal_ratio,
                    'momentpointDistance':momentpointDistance,
                    'middlepointDistance':middlepointDistance,
                    'steiner':steiner,
                    'mean_val':mean_val,
                    'approxAnzahl':approxAnzahl,
                    'solidity2':solidity2,
                    'seradity2':seradity2,
                    # 'angle':angle,
                    'minimal_ratio2':minimal_ratio2,
                    'areaRatio':areaRatio,
                    'contureDistance':contureDistance,
                    'conturenAnzahl':conturenAnzahl,
                    'rectangleLike':rectangleLike,
                    'circleLike':circleLike,
                    'keyPointsAnzahl':keyPointsAnzahl,
                    'farPoint':farPoint,
                    'type':objectType }

                self.__contureValues.append(dic)
                # print(dic)
                # print(keyValues)
                # self.__contureValues[g] = dict(sorted(self.__contureValues[g].items(), key=lambda kv: kv[0]))
                retList.append(drawing)
            except Exception as o:
                self.__errorList.append('ERROR: conture ' + str(e) + '--------------- ' + repr(o.args))
                print('jumped entry')
            g += 1
        self.__sampleImageList = retList

    def PrintValues(self):
        e = 0
        for entry in self.__contureValues:
            e = e + 1
            try:
                for key in entry:
                    print(key, '---', str(entry.get(key)))
                print('...........................')
            except Exception as f:
                self.__errorList.append(
                    'ERROR: PrintConture ' + str(e) + '---------------')
                print(repr(f))

    def PrintValuesToCSV(self, fileName = 'sampleCSV'):
        with open(fileName, 'w', encoding='UTF8', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            keyList = sorted(self.__contureValues[0].keys())
            csvwriter.writerow(keyList)
            for entry in self.__contureValues:
                line = []
                for key in keyList:
                    line.append(str(entry.get(key)))
                csvwriter.writerow(line)
        print('-------------Values CSV Done------------')

    def PrintErrors(self, all=False):
        print('Total Number of Errors = ' + str(len(self.__errorList)))
        if all:
            try:
                for entry in self.__errorList:
                    print(entry)
            except Exception as f:
                print('Errorlist not printable')
                print(repr(f))
    
    @property
    def sampleImageList(self):
        return self.__sampleImageList

    @property
    def sampleLenght(self):
        return len(self.__sampleImageList)

    @property
    def contureValues(self):
        return self.__contureValues

class Dectree():

    def frame_split(self, df, test_size_in_percent):
    
        test_size = round(test_size_in_percent/100 * len(df))

        indices = df.index.tolist()
        test_indices = random.sample(population=indices, k=test_size)

        test_df = df.loc[test_indices]
        train_df = df.drop(test_indices)
        
        return train_df, test_df

    def check_purity(self, data):
    
        label_column = data[:, -1]
        unique_classes = np.unique(label_column)

        if len(unique_classes) == 1:
            return True
        else:
            return False

    def classify_data(self, data):
    
        label_column = data[:, -1]
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

        index = counts_unique_classes.argmax()
        classification = unique_classes[index]
        
        return classification

    def get_potential_splits(self, data):
    
        potential_splits = {}
        _, n_columns = data.shape
        for column_index in range(n_columns - 1):        # excluding the last column which is the label
            potential_splits[column_index] = []
            values = data[:, column_index]
            unique_values = np.unique(values)

            for index in range(len(unique_values)):
                if index != 0:
                    current_value = unique_values[index]
                    previous_value = unique_values[index - 1]
                    potential_split = (current_value + previous_value) / 2
                    
                    potential_splits[column_index].append(potential_split)
        
        return potential_splits
    
    def split_data(self, data, split_column, split_value):
        
        split_column_values = data[:, split_column]

        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
        
        return data_below, data_above

    def calculate_entropy(self, data):
    
        label_column = data[:, -1]
        _, counts = np.unique(label_column, return_counts=True)

        probabilities = counts / counts.sum() #get durchschnitt/probability
        entropy = sum(probabilities * -np.log2(probabilities)) #entropie funktion np componentwise operation
        
        return entropy

    def calculate_overall_entropy(self, data_below, data_above):
    
        n = len(data_below) + len(data_above)
        p_data_below = len(data_below) / n
        p_data_above = len(data_above) / n

        overall_entropy =  (p_data_below * self.calculate_entropy(data_below) 
                        + p_data_above * self.calculate_entropy(data_above))
        
        return overall_entropy

    def determine_best_split(self, data, potential_splits):
    
        overall_entropy = 99
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = self.split_data(data, split_column=column_index, split_value=value)
                current_overall_entropy = self.calculate_overall_entropy(data_below, data_above)

                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value
        
        return best_split_column, best_split_value

    def make_decision_tree(self, df, counter=0, min_samples=2, max_depth=3):
    
        # datenframe beim ersten durchgang in numpy array umwandeln
        if counter == 0:
            global COLUMN_HEADERS
            COLUMN_HEADERS = df.columns
            data = df.values
        else:
            data = df           
        
        
        # klassifizieren wenn daten rein sind, baum zu groß oder zu wenig datenpunkte übrig
        if (self.check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
            classification = self.classify_data(data)
            
            return classification

        
        # recursiver part 
        else:    
            counter += 1

            # daten am splitt mit der niedrigsten entropie splitten
            potential_splits = self.get_potential_splits(data)
            split_column, split_value = self.determine_best_split(data, potential_splits)
            data_below, data_above = self.split_data(data, split_column, split_value)
            
            # sub-tree erstellen für diese tiefenstufe, teilt an splitvalue in größer und kleiner
            feature_name = COLUMN_HEADERS[split_column]
            question = str(feature_name) + " <= " + str(split_value)
            sub_tree = {question: []}
            
            # tree in linke und rechte seite aufspalten, jeweils subtrees erstellun usw
            yes_answer = self.make_decision_tree(data_below, counter, min_samples, max_depth)
            no_answer = self.make_decision_tree(data_above, counter, min_samples, max_depth)
            
            # wenn linke und rechte seite gleich klassifizieren dann abbruch, ansonsten subtree an den gesamten baum anhängen
            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)
            
            return sub_tree
    
    def classify_example(self, example, tree):
        question = list(tree.keys())[0]
        feature_name, _ , value = question.split()

        # ask question
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]

        # base case
        if not isinstance(answer, dict):
            return answer
        
        # recursive part
        else:
            residual_tree = answer
            return self.classify_example(example, residual_tree)

    def calculate_keynumbers(self, df, tree, label_name):

        df["classification"] = df.apply(self.classify_example, axis=1, args=(tree,))
        df["classification_correct"] = df["classification"] == df[label_name]
        overall_accuracy = df["classification_correct"].mean()

        label_column = df[label_name]
        unique_classes, _ = np.unique(label_column, return_counts=True)
        
        keynumbers = []

        for uc in unique_classes:
            
            tp_df = df.loc[(df[label_name] == uc) & (df["classification_correct"] == True)]
            fp_df = df.loc[(df[label_name] == uc) & (df["classification_correct"] == False)]
            tn_df = df.loc[(df[label_name] != uc) & (df["classification_correct"] == True)]
            fn_df = df.loc[(df[label_name] != uc) & (df["classification_correct"] == False)]

            true_positiv, _ = tp_df.shape
            false_positiv, _ = fp_df.shape
            true_negative, _ = tn_df.shape
            false_negative, _ = fn_df.shape

            recall = true_positiv / (true_positiv+false_negative)
            precision = true_positiv / (true_positiv+false_positiv)
            f1score = 2 * (precision * recall) / (precision + recall)

            keynumbers.append({"Label":uc, "recall":recall, "precision":precision, "f1score":f1score})
        
        
        return overall_accuracy, keynumbers

class Bayes():

    def frame_split(self, df, test_size_in_percent):
    
        test_size = round(test_size_in_percent/100 * len(df))

        indices = df.index.tolist()
        test_indices = random.sample(population=indices, k=test_size)

        test_df = df.loc[test_indices]
        train_df = df.drop(test_indices)
        
        return train_df, test_df

    def wahrscheinlichkeiten(self, df, label_column):
        table = {}

        # determine values for the label
        value_counts = df[label_column].value_counts().sort_index()
        table["class_names"] = value_counts.index.to_numpy()
        table["class_counts"] = value_counts.values

        # determine probabilities for the features
        for feature in df.drop(label_column, axis=1).columns:
            table[feature] = {}

            # determine counts
            counts = df.groupby(label_column)[feature].value_counts()
            df_counts = counts.unstack(label_column)

            # add one count to avoid "problem of rare values"
            if df_counts.isna().any(axis=None):
                df_counts.fillna(value=0, inplace=True)
                df_counts += 1

            # calculate probabilities
            df_probabilities = df_counts / df_counts.sum()
            for value in df_probabilities.index:
                probabilities = df_probabilities.loc[value].to_numpy()
                table[feature][value] = probabilities
                
        return table

    def example(self, row,bayes_table):
        
        class_estimates = bayes_table["class_counts"]
        for feature in row.index:

            try:
                value = row[feature]
                probabilities = bayes_table[feature][value]
                class_estimates = class_estimates * probabilities

            # skip in case "value" only occurs in test set but not in train set
            # (i.e. "value" is not in "lookup_table")
            except KeyError:
                continue

        index_max_class = class_estimates.argmax()
        prediction = bayes_table["class_names"][index_max_class]
        
        return prediction

    def calculate_keynumbers(self, df, bayes_table, label_name):

        df["classification"] = df.apply(self.example, axis=1, args=(bayes_table,))
        df["classification_correct"] = df["classification"] == df[label_name]
        overall_accuracy = df["classification_correct"].mean()

        label_column = df[label_name]
        unique_classes, _ = np.unique(label_column, return_counts=True)
            
        keynumbers = []

        for uc in unique_classes:
                
            tp_df = df.loc[(df[label_name] == uc) & (df["classification_correct"] == True)]
            fp_df = df.loc[(df[label_name] == uc) & (df["classification_correct"] == False)]
            tn_df = df.loc[(df[label_name] != uc) & (df["classification_correct"] == True)]
            fn_df = df.loc[(df[label_name] != uc) & (df["classification_correct"] == False)]

            true_positiv, _ = tp_df.shape
            false_positiv, _ = fp_df.shape
            true_negative, _ = tn_df.shape
            false_negative, _ = fn_df.shape

            recall = true_positiv / (true_positiv+false_negative)
            precision = true_positiv / (true_positiv+false_positiv)
            f1score = 2 * (precision * recall) / (precision + recall)

            keynumbers.append({"Label":uc, "recall":recall, "precision":precision, "f1score":f1score})
            
            
        return overall_accuracy, keynumbers