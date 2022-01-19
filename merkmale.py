import ImageHandler as ih
from ImageHandler import cv

loader = ih.ImageLoader('train\Dosenoeffner')
loader.AddDirRealPath('train\Flaschenoeffner')
loader.MakeSampleEntrys(1)
entrys = loader.sampleEntrysPath


changer = ih.ImageChanger(entrys,3,False)

print(changer.sampleLenght)
changer.FrameByWidth(500)
changer.Blur(3)
changer.Gray()
changer.ThresholdDual(200,255,cv.THRESH_BINARY, cv.THRESH_OTSU)
changer.Contures()
changer.PrintContureValues()
changer.ShowRandomImage()