import cv2, os, glob
import numpy as np

def onMouse(event, x, y, flags, param):
    global width, height, crop_size, curPos, cropArea
    curPos = (x,y)
    if event == cv2.EVENT_MOUSEMOVE and x is not None and y is not None:
        drawCropArea(*curPos)
    if event == cv2.EVENT_LBUTTONDOWN:
        if cropArea[0]<0 or cropArea[1]<0 or cropArea[2]>width or cropArea[3]>height: return    # if area is out of image size range, then return
        cropImage(*cropArea)
        drawCurPos(*curPos)

def showImage(path):
    global width, height, files, index
    image = cv2.imread(path)
    width = int(image.shape[1]/ratio)
    height = int(image.shape[0]/ratio)
    image = cv2.resize(image, (width, height))
    # put text
    cv2.putText(image, files[index], (0,20), fontFace=None, fontScale=0.5, color=(0,255,0), thickness=1)
    cv2.putText(image, "[Mouse left-click]: crop ans save", (0,40), fontFace=None, fontScale=0.5, color=(0,255,0), thickness=1)
    cv2.putText(image, "[Space]: Next image", (0,60), fontFace=None, fontScale=0.5, color=(0,255,0), thickness=1)
    cv2.putText(image, "[+/-]: Zoom in/out", (0,80), fontFace=None, fontScale=0.5, color=(0,255,0), thickness=1)
    cv2.putText(image, "[Esc]: Quit", (0,100), fontFace=None, fontScale=0.5, color=(0,255,0), thickness=1)
    # check cropped
    dirname = os.path.dirname(path)
    filename = os.path.basename(path)
    # cropped_folder = os.path.join(dirname, "cropped")
    cropped_file = os.path.join(cropped_folder, "crop_"+filename)
    if os.path.exists(cropped_file):
        cv2.putText(image, f"Already cropped at {cropped_file}", (0,120), fontFace=None, fontScale=0.5, color=(0,0,255), thickness=1)
    cv2.imshow("images", image)
    return image

def drawCropArea(x, y):
    global cropArea
    image = showImage(files[index])
    # draw crop area 
    px, py = x-int(crop_size/2), y-int(crop_size/2)
    w, h = crop_size, crop_size   # First we crop the sub-rect from the image
    if px<0 or py<0 or px+w>width or py+h>height: return    # if area is out of image size range, then return

    cropArea = (px, py, w, h)
    sub_img = image[py:py+h, px:px+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
    image[py:py+h, px:px+w] = res   # Putting the image back to its position
    cv2.putText(image, f"{crop_size}*{crop_size}", (px, py), fontFace=None, fontScale=0.5, color=(255,255, 255), thickness=1)
    cv2.imshow("images", image)

def drawCurPos(x, y):
    # draw cursor position
    image = showImage(files[index])
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(image, f"{x}, {y}", curPos, fontFace=font, fontScale=1, color=(0,255,0), thickness=2)
    cv2.imshow("images", image)   

def cropImage(x,y,w,h):
    global files, index
    path = files[index]
    basename = os.path.basename(path)
    image = cv2.imread(path)
    width = int(image.shape[1]/ratio)
    height = int(image.shape[0]/ratio)
    image = cv2.resize(image, (width, height))
    cropImage = image[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(cropped_folder, "crop_"+basename), cropImage)

def listAllImages(path, name):
    with open(os.path.join(path, name), "w") as f:
        files = glob.glob(os.path.join(path, "*.jpg"))
        for imgPath in files:
            f.write(imgPath+"\r\n")


# set path 
# cropped_folder = "images/cropped"
cropped_folder = "cropped"
images_folder = "images"
files = glob.glob(images_folder+"/*.jpg")     
listAllImages(cropped_folder, "train.txt")      

# set parameter 
index = 0
# ratio = 3
ratio = 3
width, height = 0, 0
# crop_size = 400
crop_size = 800
curPos = (0,0)
cropArea = (0,0,0,0)

# config cv2 window
cv2.namedWindow("images")
cv2.moveWindow("images", 100, 0)
cv2.setMouseCallback('images', onMouse)

showImage(files[index])
while True:
    c = cv2.waitKey(10)
    if c == 27: # ESC - quit 
        break
    elif c == 32: # Space - next image
        index += 1
        index %= len(files)
        showImage(files[index])

        # # skip cropped
        # filename = os.path.basename(files[index])
        # cropped_file = os.path.join(cropped_folder, "crop_"+filename)
        # while os.path.exists(cropped_file) and index < len(files):
        #     index += 1
        #     filename = os.path.basename(files[index])
        #     cropped_file = os.path.join(cropped_folder, "crop_"+filename)
        # showImage(files[index])
        # # end skip
        
    # elif c != -1:
        # print(c)
    if c == 61: # + zoom in
        crop_size += 10
        drawCropArea(*curPos)
    elif c == 45:   # - zoom out
        crop_size -= 10
        drawCropArea(*curPos)
