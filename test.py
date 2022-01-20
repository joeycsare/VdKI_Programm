import ImageHandler as ih
from ImageHandler import cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

loader = ih.ImageLoader('bilddaten\Dosenoeffner')
loader.AddDirRealPath('bilddaten\Flaschenoeffner')
loader.AddDirRealPath('bilddaten\Korkenzieher')
# loader.AddDirRealPath('train\Dosenoeffner')
# loader.AddDirRealPath('train\Flaschenoeffner')
# loader.AddDirRealPath('train\Korkenzieher')
# # for i in range(1):
loader.MakeSampleEntrys(300)
changer = ih.ImageChanger(loader.sampleEntrysPath, 15, False)
changer.FrameByValues(250,250)
changer.Normieren()
# changer.ShowAll()
changer.ColorspaceCorrection()
changer.Gray()
# changer.ShowAll()
changer.BilateralFilter(100,10)
# changer.ShowAll()
changer.AdaptiveCanny(edge=8)
# changer.ShowAll()
changer.Blur(3)
# changer.ShowAll()
changer.AdaptiveCanny(edge=4)
# changer.ThresholdTest()
# changer.ThresholdDualAdapt(contLenght=20)
# changer.ShowAll()
# # changer.Hold()
# changer.Contures(0,type1=cv.RETR_CCOMP,type2=cv.CHAIN_APPROX_NONE)
# changer.ShowAll()
# changer.Hold()
# changer.Revert()
# changer.Contures(1,type1=cv.RETR_TREE,type2=cv.CHAIN_APPROX_SIMPLE)
# changer.ShowAll()
# changer.Hold()
# changer.Revert()
changer.Contures(1,type1=cv.RETR_EXTERNAL,type2=cv.CHAIN_APPROX_NONE)
# changer.ShowAll()
# changer.Hold()




# changer.PrintErrors(True)
# img = changer.GetRandomImage()

loader.AddColumnsToSample(changer.contureValues)
loader.PrintSampleEntrys()
loader.DeleteColumnFromSample('name')
loader.DeleteColumnFromSample('index')
loader.DeleteColumnFromSample('path')
loader.DeleteColumnFromSample('directoryPath')
loader.DeleteColumnFromSample('assignment')
loader.PrintSampleEntrys()
loader.PrintSamplesToCSV()