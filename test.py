import ImageHandler as ih
from ImageHandler import cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

loader = ih.ImageLoader('iliasdaten\Dosenoeffner')
loader.AddDirRealPath('iliasdaten\Flaschenoeffner')
loader.AddDirRealPath('iliasdaten\Korkenzieher')
# loader.AddDirRealPath('train\Dosenoeffner')
# loader.AddDirRealPath('train\Flaschenoeffner')
# loader.AddDirRealPath('train\Korkenzieher')
# for i in range(1):
loader.MakeSampleEntrys(1)
changer = ih.ImageChanger(loader.sampleEntrysPath, 1, False)
changer.ColorspaceCorrection()
changer.ShowAll()
changer.Hold()
changer.Gray()
changer.FrameByWidth(500)
changer.ShowAll()
changer.Hold()
changer.Blur(3)
changer.ShowAll()
changer.Hold()
changer.ThresholdDual(tresh=127)
changer.ShowAll()
changer.Hold()
# changer.Revert()
# changer.Canny(False, 25, 230)
changer.Contures(0)
# changer.PrintContureValues()
changer.ShowAll()
changer.Hold()
changer.PrintErrors(True)

# changer.PrintErrors(True)
# img = changer.GetRandomImage()
# black = np.zeros((img.shape[0],img.shape[0],3))
# changer.ShowImage(img, 1)
# contours, hier = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
# cv.drawContours(black,contours,-1,(255,0,0),2)
# cv.imshow('contureimage', black)
# cv.waitKey(2000)

# newentrys = changer.contureValues
# loader.AddColumnsToSample(newentrys)
# loader.PrintSampleEntrys()
# loader.DeleteColumnFromSample('name')
# loader.DeleteColumnFromSample('path')
# loader.DeleteColumnFromSample('directoryPath')
# loader.DeleteColumnFromSample('assignment')
# loader.PrintSampleEntrys()
# loader.PrintSamplesToCSV()

# vdki_df = pd.read_csv('output.csv')
# print(vdki_df.head())
