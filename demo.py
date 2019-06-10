# Import the module
from WIDER import *
from detection import *
# Load the data
trainData = WIDER_image(r'H:\WIDER\WIDER_train\images',r'H:\WIDER\wider_face_split\wider_face_train_bbx_gt.txt')
# Train the model
model = YOLOv3_tiny(boxNum=64)
model.build()
model.train(imgClass=trainData,epoches=10,batchSize=2,stopRounds=1)
# Load the model
detector = Detector()