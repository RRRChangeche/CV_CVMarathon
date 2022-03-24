import numpy as np 

class DartBoard:
    def __init__(self):
        self.center = 0 
        self.R = 0
        self.Rscale = [i/340 for i in [12.7, 32, 182, 214, 308,  340]]
        self.calibratePoints = []
        self.scoreMap = {0:1,1:18,2:4,3:13,6:10,7:15,8:2,9:17,10:3,11:19,12:7,13:16,14:8,15:11,16:14,17:9,18:12,19:5,20:20}

    def getPerspectiveTransformbySIFT(self, img_std, img):
        pass

    def getPerspectiveTransform(self, pts, pts_std):
        pass

    def line_intersection(self, line1, line2):
        pass


def getScore():
    pass