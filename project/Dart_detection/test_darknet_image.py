from darknet import load_network, print_detections, decode_detection, detect_image
from darknet_images import *
from darknet_video import *
import cv2

cfg_path = ".\\cfg\\yolov4-tiny-dart.cfg"
data_path = ".\\cfg\\dart_local.data"
weights_path =".\\backup_weight\\yolov4-tiny-dart_final.weights"
image_path = r"C:\AL\Software\darknet-master\build\darknet\x64\data\dart\crop_PXL_20220217_132917779.jpg"

# set network
network, class_names, class_colors = darknet.load_network(
            cfg_path,
            data_path,
            weights_path,
            batch_size=1
        )

# dart detection
image, detections = image_detection(
            image_path, network, class_names, class_colors, 0.5
)

# draw dartboard scoring region lines 
calibratePoints = {}
for d in detections:
    if d[0] == 'topP': calibratePoints['topP'] = tuple(int(i) for i in d[2][:2])
    elif d[0] == 'bottomP': calibratePoints['bottomP'] = tuple(int(i) for i in d[2][:2])
    elif d[0] == 'leftP': calibratePoints['leftP'] = tuple(int(i) for i in d[2][:2])
    elif d[0] == 'rightP': calibratePoints['rightP'] = tuple(int(i) for i in d[2][:2])
    # elif d[0] == 'dartP': calibratePoints.append([d[2][:2]])

print(calibratePoints)
if len(calibratePoints) > 2:    # if there are enough cal points
    image = cv2.imread(image_path)
    image = cv2.resize(image, (416,416))
    for p in calibratePoints:
        cv2.circle(image, calibratePoints[p], 3, (0,255,0), 3)
    cv2.imshow("dart", image)

cv2.waitKey(0)

# 262,49/ 203,409/ 53,196/ 416, 254