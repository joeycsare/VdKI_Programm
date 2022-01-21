import os
import random
import csv
import cv2 as cv
import numpy as np


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
                print("no pop")
            entry = dict(sorted(entry.items(), key=lambda kv: kv[0]))

    def AddRow(self, Row={}):
        self.__Data.append(Row)

    def DeleteRow(self, RowNumber=0):
        self.__Data.pop(RowNumber)

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
                print("no pop")
            entry = dict(sorted(entry.items(), key=lambda kv: kv[0]))

    def AddRowToSample(self, Row={}):
        self.__sampleEntrys.append(Row)

    def DeleteRowFromSample(self, RowNumber=0):
        self.__sampleEntrys.pop(RowNumber)

    def PrintSamplesToCSV(self):
        with open('output.csv', 'w', encoding='UTF8', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            keyList = sorted(self.__sampleEntrys[0].keys())
            csvwriter.writerow(keyList)
            for entry in self.__sampleEntrys:
                line = []
                for key in keyList:
                    line.append(str(entry.get(key)))
                csvwriter.writerow(line)
        print('-------------CSV Done------------')

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
    __originalList = []
    __lastList = []
    __contureValues = []
    __calculatedValues = []
    __errorList = []

    @property
    def sampleImageList(self):
        return self.__sampleImageList

    @property
    def sampleLenght(self):
        return len(self.__sampleImageList)

    @property
    def contureValues(self):
        return self.__contureValues

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
                    self.__contureValues.append({})
                    self.__calculatedValues.append({})
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
                img = cv.dilate(frame, (kernelsize, kernelsize),
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

    def Erode(self, kernelsize=3, iterate=1):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = cv.erode(frame, (kernelsize, kernelsize),
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
                clahe = cv.createCLAHE(clipLimit=Limit, tileGridSize=(GridSize, GridSize))
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

    def ColorspaceToColor(self, targetColor = 0, showPrewiev=False):
        retList = []
        e = 0
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                lab = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                a, b, c = cv.split(lab)
                anew = np.zeros((frame.shape[0], frame.shape[1], 1), dtype=np.uint8)+targetColor
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
                self.__errorList.append('ERROR: ColorspaceCorrection ' + str(e) + '--------------- ' + repr(f))

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

    def CannyNorm(self, track=False, tres1=125, tres2=175):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                img = cv.Canny(frame, tres1, tres2, L2gradient=True)
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
                while len(contas) < contLenght:
                    m = np.median(frame)  # Medianwert
                    # lower threshold (Grenze)
                    lower = wide + int(max(0, (1.0 - sigma) * m))
                    # upper threshold (Grenze)
                    upper = wide + int(min(255, (1.0 + sigma) * m))
                    # OpenCV, die de Kante sucht
                    img = cv.Canny(frame, lower, upper,edges=edge ,L2gradient=True)
                    contas, hier = cv.findContours(
                        img, cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
                    wide -= 5
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

    def ThresholdTest(self,tresh=0, type1=cv.THRESH_BINARY, type2=cv.THRESH_OTSU, invert=True):
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
                T1 = cv.threshold(self.__sampleImageList[0], tresh, 255, type1, type2)
                if invert:
                    T1 = cv.threshold(self.__sampleImageList[0], T1[0], 255, cv.THRESH_BINARY_INV)
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

    def ThresholdDualAdapt(self, contLenght=5,type1=cv.THRESH_BINARY, type2=cv.THRESH_OTSU, invert=True):
        retList = []
        e = 0
        self.__lastList = self.__sampleImageList
        for frame in self.__sampleImageList:
            e = e + 1
            try:
                contas = []
                wide = 0
                while (len(contas) < contLenght):# & (wide < 255):
                    img = cv.threshold(frame, wide, 255, type1, type2)
                    contas, hier = cv.findContours(img[1], cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
                    cv.imshow('test',img[1])
                    cv.waitKey(1000)
                    wide += 5
                retList.append(frame)
            except Exception as f:
                self.__errorList.append('ERROR: Threshold adapt ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def __nothing(self, x):
        pass

    def Contures(self, deepness=1, type1=cv.RETR_EXTERNAL, type2=cv.CHAIN_APPROX_NONE, epsyValue=0.1, printing=False):
        retList = []
        self.__lastList = self.__sampleImageList
        e = 0
        g = 0
        for frame in self.__sampleImageList:  # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
            e = e + 1
            drawing = np.zeros(
                (frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            contours, _ = cv.findContours(frame, type1, type2)
            l = []
            for c in contours:
               l.append(cv.convexHull(c)) 
            sorted_contours = sorted(contours, key=cv.contourArea, reverse=False)
            print('conturanzahl = ', len(contours))
            depth = deepness
            if depth == 0:
                depth = len(contours)
            while depth > 0:
                try:
                    # if printing:
                    #     areas = []
                    #     for conts in sorted_contours:
                    #         carea = cv.contourArea(conts)
                    #         areas.append(carea)

                    #     print(areas)

                    cnt = sorted_contours[len(contours)-depth]

                except Exception as f:
                    self.__errorList.append(
                        'ERROR: Conture_Making cnt_' + str(depth) + '_' + str(e) + '---------------')
                    print(repr(f))

                try:

                    # Centroid!!!!!!!!!!!!!
                    M = cv.moments(cnt)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                except Exception as f:
                    self.__errorList.append(
                        'ERROR: Conture_Making Centroid_' + str(depth) + '_' + str(e) + '---------------')
                    print(repr(f))

                try:

                    # Perimeter and Area !!!!!!!!!!
                    perimeter = cv.arcLength(cnt, True)
                    area = cv.contourArea(cnt)
                    approxConture = cv.approxPolyDP(cnt, epsyValue*cv.arcLength(cnt, True), True)
                    approxContureLenght = cv.arcLength(approxConture, True)
                    approxContureArea = cv.contourArea(approxConture)
                    hull = cv.convexHull(cnt)
                    convexHullArea = cv.contourArea(hull)
                    convexHullLenght = cv.arcLength(hull, closed=True)

                except Exception as f:
                    self.__errorList.append(
                        'ERROR: Conture_Making Area and Perimeter_' + str(depth) + '_' + str(e) + '---------------')
                    print(repr(f))

                try:

                    # Find the rotated rectangles
                    minRect = [None]*len(sorted_contours)  # Contour !!!!!!!!!!
                    for j, c in enumerate(sorted_contours):
                        minRect[j] = cv.minAreaRect(c)

                    mx = int(minRect[len(minRect)-depth][0][0])
                    my = int(minRect[len(minRect)-depth][0][1])

                except Exception as f:
                    self.__errorList.append(
                        'ERROR: Conture_Making Rectangles_' + str(depth) + '_' + str(e) + '---------------')
                    print(repr(f))

                try:
                    cv.drawContours(drawing, sorted_contours, len(
                        sorted_contours)-depth, (255, 255, 255))  # Contour

                    box = cv.boxPoints(minRect[len(minRect)-depth])
                    box = np.intp(box)
                    # Box(lila)
                    cv.drawContours(drawing, [box], 0, (255, 0, 255))
                    # Mittelpunkt(grün) der Box
                    cv.circle(drawing, (mx, my), 3, (0, 255, 0))
                    # Schwerpunkt(rot) der Fläche der Kontur
                    cv.circle(drawing, (cx, cy), 3, (255, 0, 0))
                except Exception as f:
                    self.__errorList.append(
                        'ERROR: Conture_Making Drawing_' + str(depth) + '_' + str(e) + '---------------')
                    print(repr(f))

                try:
                    stein = np.sqrt(((mx-cx)**2)+((my-cy)**2))
                    boxLenght, boxWidth = minRect[len(minRect)-depth][1]
                    ratio = max(minRect[len(minRect)-depth][1]) / \
                        min(minRect[len(minRect)-depth][1])
                except Exception as f:
                    self.__errorList.append(
                        'ERROR: Conture_Calculating Values_' + str(depth) + '_' + str(e) + '---------------')
                    print(repr(f))

                try:
                    keyValues = [perimeter, stein, ratio, area,
                                 approxContureLenght, approxContureArea, 
                                 convexHullArea, convexHullLenght]
                    keyNames = ['perimeter', 'stein', 'ratio', 'area',
                                'approxContureLenght', 'approxContureArea', 
                                'convexHullArea', 'convexHullLenght']
                    dic = {}
                    for k in range(len(keyNames)):
                        l = keyNames[k] + '_' + str(depth)
                        dic[l] = keyValues[k]
                except Exception as f:
                    self.__errorList.append(
                        'ERROR: Conture_Making Dictionary_' + str(depth) + '_' + str(e) + '---------------')
                    print(repr(f))

                    # keyValues = [convexHullLenght/perimeter, boxLenght,boxWidth,]
                    # keyNames = ['denting', 'convexness','']
                    # calcdic = {}
                    # for k in range(len(keyNames)):
                    #     l = keyNames[k] + '_' + str(depth)
                    #     dic[l] = keyValues[k]

                try:
                    self.__contureValues[g].update(dic)
                    self.__contureValues[g] = dict(
                        sorted(self.__contureValues[g].items(), key=lambda kv: kv[0]))

                except Exception as f:
                    self.__errorList.append(
                        'ERROR: Conture_Adding Values to Main_' + str(depth) + '_' + str(e) + '---------------')
                    print(repr(f))
                depth -= 1
            g += 1
            retList.append(drawing)
        self.__sampleImageList = retList

    def PrintContureValues(self):
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

    def PrintCalculatedValues(self):
        e = 0
        for entry in self.__contureValues:
            e = e + 1
            try:
                for key in entry:
                    print(key, '---', str(entry.get(key)))
                print('...........................')
            except Exception as f:
                self.__errorList.append(
                    'ERROR: PrintCalculated ' + str(e) + '---------------')
                print(repr(f))

    def PrintErrors(self, all=False):
        print('Total Number of Errors = ' + str(len(self.__errorList)))
        if all:
            try:
                for entry in self.__errorList:
                    print(entry)
            except Exception as f:
                print('Errorlist not printable')
                print(repr(f))

    def ColorMerkmale(self, showPrewiev=False,sigma=0.9, type1=cv.THRESH_BINARY, type2=cv.THRESH_OTSU):
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

                holz = np.zeros((hsv.shape[0], hsv.shape[1], 1), dtype=np.uint8)
                rows,cols = hue.shape # holziness
                for i in range(rows):
                    for j in range(cols):
                        h,s,v = hsv[i,j]
                        if h < 70 | h > 220 & s > 50:
                            holz[i,j] = 250

                bright = cv.bitwise_not(bright) # brightness invertieren
                bright2 = cv.add(bright, sat) # brightness auf sättigung addieren
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
                self.__errorList.append('ERROR: ColorspaceCorrection ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList

    def BrightTresh(self, showPrewiev=False,sigma=0.9, type1=cv.THRESH_BINARY, type2=cv.THRESH_OTSU):
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
                self.__errorList.append('ERROR: BrightTresh ' + str(e) + '--------------- ' + repr(f))
        self.__sampleImageList = retList
