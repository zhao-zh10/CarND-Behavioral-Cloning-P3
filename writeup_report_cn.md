# **驾驶行为克隆**

---

**驾驶行为克隆项目**

项目目标/步骤如下 :
* 使用模拟器收集良好的驾驶行为数据
* 使用Keras构建卷积神经网络，根据输入图像预测车辆转向角
* 使用训练集和验证集进行训练和交叉验证模型
* 测试模型是否能在测试路段1中成功驾驶，不驶出路面
* 测试模型是否能在测试路段2中成功驾驶，不驶出路面
* 总结项目实验结果


[//]: # (Image References)

[image_model_visualization]: ./examples/model_visualization.png "Model Visualization"
[image_training_history]: ./examples/history.png "Model Training History"
[image_angle_distribution_total]: ./examples/Angle_Data_Distribution_Total.png "Angle Data Distribution (Total)"
[image_angle_distribution_filtered]: ./examples/Angle_Data_Distribution_Filtered.png "Angle Data Distribution (Filtered)"
[image_center_lane_track1]: ./examples/center_lane_track1.jpg "Center Lane Track 1"
[image_center_lane_track1_counter-clockwise]: ./examples/center_lane_track1_counter-clockwise.jpg "Center Lane Track 1 Counter Clockwise"
[image_center_lane_track2]: ./examples/center_lane_track2.jpg "Center Lane Track2"
[image_center_lane_track1_flipped]: ./examples/center_lane_track1_flipped.jpg "Center Lane Track1 Flipped"
[image_left_camera]: ./examples/left_camera.jpg "Left Camera Image"
[image_central_camera]: ./examples/central_camera.jpg "Central Camera Image"
[image_right_camera]: ./examples/right_camera.jpg "Right Camera Image"
[image_using_multiple_cameras]: ./examples/carnd-using-multiple-cameras.png "Using Multiple Cameras"
## 完成项目要点
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py----containing the script to create and train the model
* drive.py----for driving the car in autonomous mode
* model.h5----containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf----summarizing the results
* video_track1.mp4----A video recording of my vehicle driving autonomously at least one lap around the track one.
* video_track2.mp4----A video recording of my vehicle driving autonomously at least one lap around the track two.

#### 2. Submission includes functional code
Using the [Udacity provided simulator](https://github.com/udacity/self-driving-car-sim) and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I referred to Nvidia's convolutional neural network for self-driving cars from the paper--[End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316v1) and [Comma.ai's steering angle prediction model](https://github.com/commaai/research/blob/master/train_steering_model.py).
My model consists of a convolution neural network based on Nvidia's model architecture(model.py lines 124-145) but slightly different: I add a Cropping2D layer to crop the input image and I add some dropout layers to generalize the model better.

The model includes RELU layers to introduce nonlinearity (code line 130-134, 137, 139, 141), and the data is normalized and mean centered in the model using a Keras lambda layer (code line 129).

For details about the final chosen appropriate model, see the next section(Final Model Architecture).

#### 2. Attempts to reduce overfitting in the model

The model contains 4 dropout layers in order to reduce overfitting (model.py lines 136, 138, 140, 142).

The model was trained and validated on different data sets(both driving data for track 1 and track 2) to ensure that the model was not overfitting (code line 161-169).

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was tuned automatically (model.py line 173).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, driving in a counter-clockwise direction, and using both track 1 and track 2 driving behavior data.

For data augmentation, I also used multiple cameras' images and flipped the images and steering measurements.

I filtered about 90% driving data samples with 0 degree steering angle, because they were overrepresented and made the datasets imbalanced.

For details about how I created the training data, see the next section(Creation of the Training Set & Training Process).

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use CNN (short for convolutional neural network) and followd by a few fully connected layers. The Nvidia's paper and Comma.ai's steering angle prediction model mentioned above were good referrence on this topic.

My first step was to use a convolution neural network model similar to the Nvidia's model, I thought this model might be appropriate because they have succeeded on real cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by using some dropout layers between fully connected layers.

The final step was to run the simulator to see how well the car was driving around both track one and track two. I found there was a left turn bias on track one, so I collected counter-clockwise laps data around the track one and used image flipping technique to overcome this obstacle.

At the end of the process, the vehicle is able to drive autonomously around both the track one and track two without leaving the road.

#### 2. 最终模型结构

最终模型结构(见代码model.py 第 124-145 行)为一卷积神经网络，每层尺寸如下所示：

model = Sequential()  
model.add(Cropping2D(cropping=((50, 20), (0, 0)),input_shape=(160, 320, 3)))   
model.add(Lambda(lambda x: x / 127.5 - 1.0))  
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))  
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))  
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))  
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))  
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))  
model.add(Flatten())  
model.add(Dropout(0.2))  
model.add(Dense(100, activation='relu'))  
model.add(Dropout(0.3))  
model.add(Dense(50, activation='relu'))  
model.add(Dropout(0.3))  
model.add(Dense(10, activation='relu'))  
model.add(Dropout(0.2))  
model.add(Dense(1))  

模型概要信息如下(由Keras输出):  

|网络层 (类型)           |          输出尺寸     |     参数数量    |                     
|:------------------------------:|:--------------------------:|:--------------:|
| cropping2d_1 (Cropping2D)      |    (None, 90, 320, 3)      |   0            |
| Normalization (Lambda)         |    (None, 90, 320, 3)      |   0            |
| conv2d_1 (Conv2D)              |    (None, 43, 158, 24)     |   1824         |   
| conv2d_2 (Conv2D)              |    (None, 20, 77, 36)      |   21636        |  
| conv2d_3 (Conv2D)              |    (None, 8, 37, 48)       |   43248        |  
| conv2d_4 (Conv2D)              |    (None, 3, 18, 64)       |   27712        |
| conv2d_5 (Conv2D)              |    (None, 1, 8, 64)        |   36928        |
| flatten_1 (Flatten)            |    (None, 512)             |   0            |   
| dropout_1 (Dropout)            |    (None, 512)             |   0            |
| dense_1 (Dense)                |    (None, 100)             |   51300        |
| dropout_2 (Dropout)            |    (None, 100)             |   0            |  
| dense_2 (Dense)                |    (None, 50)              |   5050         |
| dropout_3 (Dropout)            |    (None, 50)              |   0            |   
| dense_3 (Dense)                |    (None, 10)              |   510          |
| dropout_4 (Dropout)            |    (None, 10)              |   0            |  
| dense_4 (Dense)                |    (None, 1)               |   11           |

参数总数量: 188,219 ;  可训练参数数量: 188,219 ;  不可训练参数数量: 0


下图是由Google的[TensorBoard](https://github.com/tensorflow/tensorboard) 库生成的模型结构可视化图.

![alt text][image_model_visualization]

#### 3. 创建训练集以及训练过程

为了获取良好驾驶行为数据，首先录制测试路段1中在紧靠车道中央行驶三圈的驾驶数据。下面是在紧靠车道中央行驶的图像举例：

![alt text][image_center_lane_track1]

然后录制测试路段1中逆时针方向驾驶四圈的驾驶数据。因为在测试路段1中行驶，数据中会产生左转的偏差（bias），如果仅仅只在测试路段1中顺时针方向驾驶，模型训练数据是有偏的，会有左转的偏差。逆时针方向驾驶是一种克服这种偏差的方式，也类似于给模型一个新的测试路段的学习数据，更有助于模型的泛化。下图为逆时针行驶的摄像头记录图像举例:

![alt text][image_center_lane_track1_counter-clockwise]

然后我在测试路段2中重复上述过程，采集更多的数据点，可以更好地泛化端到端神经网络模型。下图为在测试路段2中紧靠车道中央行驶摄像头记录图像举例:

![alt text][image_center_lane_track2]

为了增加数据集，采用左右翻转摄像头采集图像以及转向角的方法，这将有助于克服左转的偏差。例如，下图是一副使用该方法增加数据的例子（使用numpy.fliplr() 函数来左右翻转图像，见代码model.py 第94-107行):
```
import numpy as np
image_flipped = np.fliplr(image_original)
steering_angle_flipped = -steering_angle
```
原始图像示例   
![alt text][image_center_lane_track1]    
左右翻转图像示例    
![alt text][image_center_lane_track1_flipped]

I also used multiple cameras' images. The simulator captures images from three cameras mounted on the car: a center, right and left camera. The following image shows a bird's-eye perspective of the car. From the perspective of the left camera, the steering angle would be less than the steering angle from the center camera. And from the right camera's perspective, the steering angle would be larger than the angle from the center camera(model.py lines 71-90).
![alt text][image_using_multiple_cameras]
I chose ```steering_correction=0.2```, and used following code segments to calculate multiple cameras' steering angles.
```
left_angle = center_angle + steering_correction
right_angle = center_angle - steering_correction
```
Left Camera Image Example  
![alt text][image_left_camera]  
Central Camera Image Example  
![alt text][image_central_camera]  
Right Camera Image Example  
![alt text][image_right_camera]  

After the collection process, I had 14,937 track one driving data samples and 30,184 track two data samples(before using multiple cameras and image flipping technique to augment the driving dataset). The distribution of steering angles is shown below:
![alt text][image_angle_distribution_total]  

From the figure above, we can figure out the the driving data samples with 0 degree steering angle were overrepresented, and the dataset is too imbalanced. So I filtered about 90% driving data samples with 0 degree steering angle(model.py lines 32-39). After that, the distribution of steering angles is shown below:
![alt text][image_angle_distribution_filtered]

I finally randomly shuffled the data set and put 20% of the data into a validation set and use generator function to feed data into the model.(model.py lines 166-169)

I used this training data for training the model. The validation set helped determine if the model was overfitting or underfitting. The ideal number of epochs was 16 as evidenced by the following image which showed the training history. I used an adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image_training_history]
