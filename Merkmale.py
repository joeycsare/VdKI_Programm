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

# loader.AddDirRealPath('iliasdaten\Dosenoeffner')
# loader.AddDirRealPath('iliasdaten\Flaschenoeffner')
# loader.AddDirRealPath('iliasdaten\Korkenzieher')

loader.MakeSampleEntrys(760)

print('dateilänge = ',len(loader.allImgPaths))
print('Randomlänge = ',len(loader.sampleEntrysPath))


changer = ih.ImageChanger(loader.sampleEntrysPath, 0, False)
changer.FrameByWidth(400)
changer.BilateralFilter(10,10)
# changer.ShowAll()
changer.Normieren()
changer.ColorMerkmale(sigma=1)
# changer.ShowAll()
changer.BrightTresh()
# changer.ShowAll()
changer.Bordering(100)
changer.Dilation(7,2)
changer.Erode(7,2)
# changer.CannyTest()
changer.ConturMerkmale(type1=cv.RETR_TREE,type2=cv.CHAIN_APPROX_NONE,epsyValue=0.005)
# changer.ShowAll()
changer.PrintErrors(all=True)
# changer.PrintValues()
changer.PrintValuesToCSV('extracsvAll')