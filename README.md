YOLOv3-tiny-FaceDetection-by-PaddlePaddle
===
![YOLOv3-tiny](https://github.com/wudejian789/YOLOv3-tiny-FaceDetection-by-PaddlePaddle/blob/master/YOLOv3-tiny.png)
>Note: If there's any error like "No module named 'xxx'", please use command "pip install xxx" to repair.   

Feel free to add my QQ: ***793729558*** to discuss with me.  
Also you can add the QQ group: ***647303915*** to discuss together.  

# 1. Import the module
```python
from imageSolver import *
from detection import *
```
# 2. How to load the data
```python
trainData = WIDER(imgPath='xxx/images',
                  imgBboxGt='xxx/wider_face_train_bbx_gt.txt'
                  maxBoxNum=256)
```
>***imgPath*** is your img root path;  
>***imgBboxGt*** is your "..._bbx_gt.txt" path;  
>***maxBoxNum*** is the maximum number of faces in each picture;  

The storage structure of image data is based on WIDER data.   
If you want to use your own data, please keep storage structure consistent or rewrite the data utils class.  
# 3. How to train your model
First you need to create a YOLOv3_tiny object.  
```python
model = YOLOv3_tiny(boxNum=64, use_cuda=True)
``` 
>***use_cuda*** is whether to use GPU;  

Then you need to build the model structure.  
```python
model.build(boxNum=64)
```
>***boxNum*** is the maximum number of faces in each picture which should be consistent with WIDER's ***maxBoxNum***.    

Next you can train your model.  
```python
model.train(imgClass=trainData, epoches=10, batchSize=64)
```
>***imgClass*** is your img data class object.  
>***batchSize*** is the number of data used for each train step;  
>***epoch*** is the total iteration number of your training data;  

And the log will be print like follows:  
```
...
After iterations 7400: loss = 52.402  mAP: 0.478  51.425 images//s  Estimated remaining time: 57995.182s
After iterations 7500: loss = 32.706  mAP: 0.575  51.565 images//s  Estimated remaining time: 57775.328s
After iterations 7600: loss = 17.977  mAP: 0.511  51.344 images//s  Estimated remaining time: 57961.978s
...
```
Finally you need to save your model for future use.  
```
model.save('./infer_model')
```
>First parameter is the path of model saved.  

Ok, I know you are too lazy to train your own model. Also you can use my trained model in `'/infer_model'`.  The model is trained on WIDER train data and score ***mAP = 0.779*** on WIDER val data.  
# 4. How to use your model to detect face
First you need to create a Detector object.  
```python
detector = Detector(modelPath='./infer_model',USE_CUDA=False)
```
>***modelPath*** is your model path;  
>***USE_CUDA*** is whether to use GPU;  

Then you can do face detection by call it.  
```python
imgs,bboxes_pre = detector(imgList=['imgs/1.jpg','imgs/2.png'],
                          confidence_threshold=0.5,nms_threshold=0.3)
```
>***imgList*** is your image path list;
>***confidence_threshold***  is confidence threshold in non-maximum suppression;  
>***nms_threshold*** is nms threshold in non-maximum suppression;  

It will return a list of each image and its detection result.  
Also you can plot it and save result:  
```
for i,(img,bbox_pre) in enumerate(zip(imgs,bboxes_pre)):
    draw_bbox(img, bbox_pre, savePath=f'imgs/{i+1}_out.png')
``` 
>***savePath*** is your image save path;  

The results are shown below:  
![result](https://github.com/wudejian789/YOLOv3-tiny-FaceDetection-by-PaddlePaddle/blob/master/imgs/res1.png)  
![result](https://github.com/wudejian789/YOLOv3-tiny-FaceDetection-by-PaddlePaddle/blob/master/imgs/res2.png)
![result](https://github.com/wudejian789/YOLOv3-tiny-FaceDetection-by-PaddlePaddle/blob/master/imgs/res3.png)
![result](https://github.com/wudejian789/YOLOv3-tiny-FaceDetection-by-PaddlePaddle/blob/master/imgs/res4.png)
![result](https://github.com/wudejian789/YOLOv3-tiny-FaceDetection-by-PaddlePaddle/blob/master/imgs/res5.png)