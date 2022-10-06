# cv-marathon
[Copuy 課程網頁](https://www.cupoy.com/marathon/0000017705882449000000016375706F795F72656C656173654355)

[Project 原文連結](https://www.cupoy.com/post/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753)

## 一、專題摘要
### 期末專題主題：飛鏢即時分數辨識
### 期末專題基本目標：
在沒有電子標靶的情況下，利用一台相機搭配傳統標靶，希望做到在任何角度可以即時辨識飛鏢落點判斷得分。參考論文實作將關鍵點偵測轉化為物件偵測問題，使用自己蒐集的資料集先進行預處理、標註、產生yolo格式的標註檔，分為４個校正點＋飛鏢落點共５個物件分類，接著使用darknet在colab中訓練模型，最後使用電腦視覺技巧校正標靶和繪製得分區域，計算飛鏢落點得分。

## 二、實作方法介紹
### 資料來源
以自己收集的資料最為資料集進行訓練，拍攝內容為每局３隻飛鏢射完後拍攝兩張不同角度的照片，確保可以辨識飛鏢落點，拍攝約１００張照片後進行資料處理
- 拍攝空標靶照片，正視
- 拍攝飛鏢射中後照片數張、各種角度以供訓練

### 資料處理
使用開源標註軟體labelImg (https://github.com/tzutalin/labelImg) 進行標註並輸出yolo格式標註檔標註流程如下：

1. 先將資料前處理每張照片裁切為以標靶為中心的正方形圖片https://github.com/RRRChangeche/CV_CVMarathon/blob/342d4d158b39e4a60fd6b4478478fc97db465d71/project/cropImages.py
2. 使用labelimg分為五種類別標出4個校正點位置及飛鏢位置，做object detection，分別為topP/bottomP/leftP/rightP/dartP
3. 輸出為yolo格式的標註檔，分別存在yolo-train和yolo-val資料夾內，資料夾內每張圖片就會有一個yolo格式的標註檔 (.txt) 格式為
```
[category number] [obejct center in X] [object center in Y] [object width in X] [object height in Y]
```
![2_1.png](http://clubfile.cupoy.com/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753/1643188284908/large)
![2_2.png](http://clubfile.cupoy.com/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753/1643188284907/large)

### 訓練模型
資料整理及標註完後可以開始訓練模型，本文將使用yolov4-tiny model模型訓練，在訓練模型前也有幾項前置工作需要處理，如下
- 下載darknet並編譯
- 編輯設定檔
- 訓練模型

### 下載darknet並編譯
從darknet github頁面下載原始碼後編譯
https://github.com/AlexeyAB/darknet

### 編輯設定檔  
在專案資料夾下新增參數資料夾(cfg)，共有五個設定檔需要編輯包括
dart.names、dart.data、yolov4-tiny.cfg、train.txt、val.txt
1. 先從"...\darknet-master\cfg" 資料夾內複製 coco.names、coco.data、yolov4-tiny-custom.cfg 到新創建的cfg資料夾中

2. dart.names: 內容為標籤的列表
   將coco.names改名為 dart.names，照格式新增需要的物件標籤名稱如下
```
topP
bottomP
leftP
rightP
dartP
```

2. dart.data: 定義label數量、各設定檔和權重檔路徑
   coco.data改名為dart.data，並照格式修改為修改為如下
```
classes= 5
train  = /content/drive/MyDrive/Colab_Notebooks/Dart_detection/cfg/train.txt
valid  = /content/drive/MyDrive/Colab_Notebooks/Dart_detection/cfg/val.txt
names = /content/drive/MyDrive/Colab_Notebooks/Dart_detection/cfg/dart.names
backup = /content/drive/MyDrive/Colab_Notebooks/Dart_detection/backup_weight/
eval=coco
```

3. yolov4-tiny-custom.cfg: 模型的設定檔，定義模型各種參數
   subdivision為每個batch要拆成幾組訓練
   max_batches 官網有公式 classes*2000
   steps 官網有公式 80% and 90% of max_batches
   tiny model有兩組detector所以更改兩組detector的filters/ anchors/ classes
```
Line 7  : subdivisions=64
Line 20 : max_batches = 10000 
Line 22 : steps=8000,9000
Line 212: filters=30 
Line 219: anchors = 31, 31,  37, 37,  41, 41,  42, 42,  42, 42,  52, 52 
Line 220: classes=80
Line 263: filters=30
Line 268: anchors = 31, 31,  37, 37,  41, 41,  42, 42,  42, 42,  52, 52
Line 269: classes=5
```
(詳細流程也可以參考 darknet github 官方頁面 How to train (to detect your custom objects https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)

4. train.txt、val.txt: 內容為處理完後訓練集和驗證集的圖片路徑
```
/content/drive/MyDrive/Colab_Notebooks/Dart_detection/yolo-train/crop_PXL_20220217_131953310.jpg
/content/drive/MyDrive/Colab_Notebooks/Dart_detection/yolo-train/crop_PXL_20220217_132530498.jpg
/content/drive/MyDrive/Colab_Notebooks/Dart_detection/yolo-train/crop_PXL_20220217_132746735.jpg
/content/drive/MyDrive/Colab_Notebooks/Dart_detection/yolo-train/crop_PXL_20220217_132917779.jpg 
...
```
編輯完設定檔後，cfg目錄下總共有這些檔案
![2_3.png](http://clubfile.cupoy.com/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753/1643188285087/large)

### 訓練模型
使用colab將所有檔案上傳後
1. 在colab環境編譯darknet
```
!git clone https://github.com/AlexeyAB/darknet.git 
%cd darknet
!pwd 
!sed -i "s/GPU=0/GPU=1/g" ./Makefile 
!sed -i "s/CUDNN=0/CUDNN=1/g" ./Makefile 
!sed -i "s/CUDNN_HALF=0/CUDNN_HALF=1/g" ./Makefile 
!sed -i "s/OPENCV=0/OPENCV=1/g" ./Makefile 
!head Makefile
!make 
!pwd
```

2. 訓練模型
colab訓練可以會因為超出記憶體或是超出運行時間訓練被打斷，可以利用checkpoint備份的權重檔重新訓練(backup_weight) https://pjreddie.com/darknet/yolo/
```
# 訓練模型 
!./darknet detector train \ 
/content/drive/MyDrive/Colab_Notebooks/Dart_detection/cfg/dart.data \ 
/content/drive/MyDrive/Colab_Notebooks/Dart_detection/cfg/yolov4-tiny-dart.cfg \ 
/content/drive/MyDrive/Colab_Notebooks/Dart_detection/backup_weight/yolov4-tiny-dart_last.weights -dont_show
```

### 模型應用
在python檔案裡使用darknet做object detection
1. 載入訓練好的神經網路
	- 使用 darknet.py裡的load_network function載入訓練好的網路
```
load_network
```
	- 使用darknet_images.py裡的 image_detection function預測
```
image_detection
```
	- 得到4個校正點 {topP, rightP, bottomP, leftP} 飛鏢頭尖點位置 {dartP}

2. 判斷標靶位置
	- 給訂標準標靶照片及標準校正點做比對
	- 使用opencv判斷標靶位置並裁切
	- 校正透視變形 Calibrate dartboard
		- ![2_5.png](http://clubfile.cupoy.com/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753/1643188285088/large)
		- 使用關鍵點位置 SIFT
			- cv2.SIFT_create
			- cv2.FlannBasedMatcher
			- cv2.findHomography
			- cv2.warpPerspective
		- 或使用4個校正點計算
			- cv2.getPerspectiveTransform
			- cv2.warpPerspective
			  ![2_6.png](http://clubfile.cupoy.com/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753/1643188285089/large)![2_7.png](http://clubfile.cupoy.com/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753/1643188285090/large)
			  
3. 繪製得分區域
	- 透過4個校正點先算出標靶中心center/ 及半徑R
	- 定義出各個得分區域並畫出, R比例=[12.7, 32, 182, 214, 308,  340] https://www.dimensions.com/element/dartboard
	![2_8.png](http://clubfile.cupoy.com/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753/1643188285091/large)
	  

4. 判斷飛鏢落靶點
	- 透過偵測到的飛鏢頭尖點位置 {dartP} 計算落點分數
	- ![2_9.png](http://clubfile.cupoy.com/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753/1643188285092/large)
	

## 三、成果展示
### 模型評估
利用darknet中的指令評估預測結果mAP
- IOU = 0.5, mAP = 89.37%
- IOU = 0.75, mAP = 80.00%
```
darknet.exe detector map data/dart.data cfg/yolov4-tiny-dart.cfg backup/yolov4-tiny-dart_final.weights -iou_thresh 0.5 
pause
```
![3_1.png](http://clubfile.cupoy.com/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753/1643188285093/large)
![3_2.png](http://clubfile.cupoy.com/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753/1643188285095/large)
![3_3.png](http://clubfile.cupoy.com/000001806A0DD9310000000B6375706F795F72656C656173654B5741535354434C55424E455753/1643188285094/large)
從mAP結果來看，當IOU設為0.5時結果較好

## 四、結論
### 問題探討
訓練過程中可能會碰到許多問題如下：
- 在colab中訓練常常遇到GPU記憶體不足或是GPU使用時間超時還有網路不穩等問題中斷訓練，colab有時還會將你的GPU ban一段時間，遇到這種問題目前採取2個colab帳號輪流訓練並使用checkpoint之前儲存的權重接續訓練，不用重頭來過
- 因為視角問題兩隻飛鏢重疊在一起，又因為是單視角所以導致視覺無法判斷分數，可能需要改變飛鏢落點特徵框選的策略，或是使用傳統影像處理方法計算
- 有時飛鏢落在得分區的邊邊角角位置，因為一些計算誤差導致判斷得分錯誤

### 優化策略
- 針對特定情境如重疊或邊界case增加資料集和使用Data augmentation
- 未來希望根據DeepDarts參考論文中蒐集資料方法針對不同光線情況、不同遊戲確保飛鏢分布均勻
- 測試修改anchors效果
- 根據darknet官方建議優化訓練 https://github.com/AlexeyAB/darknet#how-to-improve-object-detection

### 參考資料
* 建立自己的YOLO辨識模型 – 以柑橘辨識為例
https://chtseng.wordpress.com/2018/09/01/%E5%BB%BA%E7%AB%8B%E8%87%AA%E5%B7%B1%E7%9A%84yolo%E8%BE%A8%E8%AD%98%E6%A8%A1%E5%9E%8B-%E4%BB%A5%E6%9F%91%E6%A9%98%E8%BE%A8%E8%AD%98%E7%82%BA%E4%BE%8B/
* Feature Matching + Homography to find Objects
https://docs.opencv.org/4.x/d1/de0/tutorial_py_feature_homography.html
* Deepdarts
https://github.com/wmcnally/deep-darts
https://arxiv.org/abs/2105.09880
* AlexeyAB - darknet (Github)
https://github.com/AlexeyAB/darknet
* Dartboard specification
https://www.dimensions.com/element/dartboard

## 五、期末專題作者資訊
個人Github連結：
https://github.com/RRRChangeche/CV_CVMarathon/tree/main/project
個人在百日馬拉松顯示名稱：RRR
