import numpy as np 


def getDegreeFrom2Vector(v0, v1, d0=None, d1=None):
    if not d0: d0 = np.linalg.norm(v0) 
    if not d1: d1 = np.linalg.norm(v1)
    dot = np.dot(v0, v1)
    cross = np.cross(v1, v0)
    # https://www.w3resource.com/python-exercises/numpy/linear-algebra/numpy-linear-algebra-exercise-3.php
    # if cross < 0, means vector located at 3 or 4 Quadrant
    # arccos = 0 ~ pi
    if cross < 0: theta = 360-np.arccos(dot/(d0*d1))/np.pi*180
    else: theta = np.arccos(dot/(d0*d1))/np.pi*180
    return theta

def dist(p1, p2): 
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**0.5

def polar2xy(center, r, theta):
    # return (round(center[0] + r*np.cos(theta*np.pi/180.0)), round(center[1] + r*np.sin(theta*np.pi/180.0)))
    return (round(center[0] + r*np.cos(theta*np.pi/180.0-0.5*np.pi)), round(center[1] + r*np.sin(theta*np.pi/180.0-0.5*np.pi)))


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