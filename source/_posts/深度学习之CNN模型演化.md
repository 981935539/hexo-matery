---
title: 深度学习之CNN模型演化
top: false
cover: false
toc: true
mathjax: true
date: 2019-12-30 00:47:24
password:
summary:
tags:
- DL
- Python
categories: 深度学习
---



#  前沿



# 一、LeNet

1998年LeCun发布了[LeNet](http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf)网络架构，从而揭开了深度学习的神秘面纱。

​	![](c1.png)



和“现在的CNN”相比， LeNet有几个不同点。

- 第一个不同点在于激活函数。 LeNet中使用sigmoid函数，而现在的CNN中主要使用ReLU函数。

- 第二个不同点在于池化层。原始的LeNet中使用子采样（subsampling）缩小中间数据的大小，而现在的CNN中Max池化是主流。

```python
import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

#load the MNIST dataset from keras datasets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Process data
X_train = X_train.reshape(-1, 28, 28, 1) # Expend dimension for 1 cahnnel image
X_test = X_test.reshape(-1, 28, 28, 1)  # Expend dimension for 1 cahnnel image
X_train = X_train / 255 # Normalize
X_test = X_test / 255 # Normalize

#One hot encoding
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

#Build LetNet model with Keras
def LetNet(width, height, depth, classes):
    # initialize the model
    model = Sequential()

    # first layer, convolution and pooling
    model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(5, 5), filters=6, strides=(1,1), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second layer, convolution and pooling
    model.add(Conv2D(input_shape=(width, height, depth), kernel_size=(5, 5), filters=16, strides=(1,1), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Fully connection layer
    model.add(Flatten())
    model.add(Dense(120,activation = 'tanh'))
    model.add(Dense(84,activation = 'tanh'))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model

LetNet_model = LetNet(28,28,1,10)
LetNet_model.summary()
LetNet_model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])

#Strat training
History = LetNet_model.fit(X_train, y_train, epochs=5, batch_size=32,validation_data=(X_test, y_test))

#Plot Loss and accuracy
import matplotlib.pyplot as plt
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.show()
```





# 二、AlexNet

2012年，Alex Krizhevsky发表了AlexNet，相对比LeNet它的网络层次更加深，从LeNet的5层到AlexNet的8层，更重要的是AlexNet还赢得了2012年的ImageNet竞赛的第一。AlexNet不仅比LeNet的神经网络层数更多更深，并且可以学习更复杂的图像高维特征。

​	![](c2.png)

AlexNet叠有多个卷积层和池化层，最后经由全连接层输出结果。虽然结构上AlexNet和LeNet没有大的不同，但有以下几点差异。

- 激活函数使用ReLU。
- 使用进行局部正规化的LRN（Local Response Normalization）层
- 使用Dropout
- 引入max pooling

```python
import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import Adam

#Load oxflower17 dataset
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
x, y = oxflower17.load_data(one_hot=True)

#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

#Data augumentation with Keras tools
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

#Build AlexNet model
def AlexNet(width, height, depth, classes):
    
    model = Sequential()
    
    #First Convolution and Pooling layer
    model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(width,height,depth),padding='valid',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
    #Second Convolution and Pooling layer
    model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
    #Three Convolution layer and Pooling Layer
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    
    #Fully connection layer
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.5))
    
    #Classfication layer
    model.add(Dense(classes,activation='softmax'))

    return model
  
AlexNet_model = AlexNet(224,224,3,17)
AlexNet_model.summary()
AlexNet_model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])

#Start training using dataaugumentation generator
History = AlexNet_model.fit_generator(img_gen.flow(X_train*255, y_train, batch_size = 16),
                                      steps_per_epoch = len(X_train)/16, validation_data = (X_test,y_test), epochs = 30 )

#Plot Loss and Accuracy
import matplotlib.pyplot as plt
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.show()
```



# 三、Network-in-network

2013年年尾，Min Lin提出了在卷积后面再跟一个1x1的卷积核对图像进行卷积，这就是[Network-in-network](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1312.4400)的核心思想了。NiN在每次卷积完之后使用，目的是为了在进入下一层的时候合并更多的卷积特征，减少网络参数、同样的内存可以存储更大的网络。

**1x1卷积核的作用**

- 缩放通道的大小

  通过控制卷积核的数量达到通道数大小的放缩。而池化层只能改变高度和宽度，无法改变通道数。

- 增加非线性

  1×1卷积核的卷积过程相当于全连接层的计算过程，并且还加入了非线性激活函数，从而可以增加网络的非线性，使得网络可以表达更加复杂的特征。

- 减少参数

  在Inception Network中，由于需要进行较多的卷积运算，计算量很大，可以通过引入1×1确保效果的同时减少计算量。



# 四、VGG

VGG 在 2014 年的ILSVRC比赛中最终获得了第 2 名的成绩.

​	![](c3.png)

[VGG](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1409.1556)的创新是使用3x3的小型卷积核连续卷积。重复进行“卷积层重叠2次到4次，再通过池化层将大小减半”的处理，最后经由全连接层输出结果。

​	![](c4.png)

```python

import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout
from keras.optimizers import Adam

#Load oxflower17 dataset
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
x, y = oxflower17.load_data(one_hot=True)

#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

#Data augumentation with Keras tools
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

#Build VGG16Net model
def VGG16Net(width, height, depth, classes):
    
    model = Sequential()
    
    model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(224,224,3),padding='same',activation='relu'))
    model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(17,activation='softmax'))
    
    return model
  
VGG16_model = VGG16Net(224,224,3,17)
VGG16_model.summary()
VGG16_model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])

#Start training using dataaugumentation generator
History = VGG16_model.fit_generator(img_gen.flow(X_train*255, y_train, batch_size = 16),
                                      steps_per_epoch = len(X_train)/16, validation_data = (X_test,y_test), epochs = 30 )

#Plot Loss and Accuracy
import matplotlib.pyplot as plt
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.show()
```



# 五、GoogLeNet

2014年，在google工作的Christian Szegedy为了找到一个深度神经网络结构能够有效地减少计算资源，于是有了这个[GoogleNet](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1409.4842)了（也叫做Inception V1）。在 2014 年的ILSVRC比赛中最终获得了第 1名的成绩.

​	![](c6.png)

​	![](c5.png)

GoogLeNet的特征:

-  Inception结构使用了多个大小不同的滤波器（和池化），最后再合并它们的结果
- 最重要的是使用了1×1卷积核（NiN）来减少后续并行操作的特征数量。这个思想现在叫做“bottleneck layer”。

```python

import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout,BatchNormalization,AveragePooling2D,concatenate,Input, concatenate
from keras.models import Model,load_model
from keras.optimizers import Adam

#Load oxflower17 dataset
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
x, y = oxflower17.load_data(one_hot=True)

#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

#Data augumentation with Keras tools
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

#Define convolution with batchnromalization
def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x
  
#Define Inception structure
def Inception(x,nb_filter_para):
    (branch1,branch2,branch3,branch4)= nb_filter_para
    branch1x1 = Conv2D(branch1[0],(1,1), padding='same',strides=(1,1),name=None)(x)

    branch3x3 = Conv2D(branch2[0],(1,1), padding='same',strides=(1,1),name=None)(x)
    branch3x3 = Conv2D(branch2[1],(3,3), padding='same',strides=(1,1),name=None)(branch3x3)

    branch5x5 = Conv2D(branch3[0],(1,1), padding='same',strides=(1,1),name=None)(x)
    branch5x5 = Conv2D(branch3[1],(1,1), padding='same',strides=(1,1),name=None)(branch5x5)

    branchpool = MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branchpool = Conv2D(branch4[0],(1,1),padding='same',strides=(1,1),name=None)(branchpool)

    x = concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)

    return x
  
#Build InceptionV1 model
def InceptionV1(width, height, depth, classes):
    
    inpt = Input(shape=(width,height,depth))

    x = Conv2d_BN(inpt,64,(7,7),strides=(2,2),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)
    x = Conv2d_BN(x,192,(3,3),strides=(1,1),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)

    x = Inception(x,[(64,),(96,128),(16,32),(32,)]) #Inception 3a 28x28x256
    x = Inception(x,[(128,),(128,192),(32,96),(64,)]) #Inception 3b 28x28x480
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x) #14x14x480

    x = Inception(x,[(192,),(96,208),(16,48),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(160,),(112,224),(24,64),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(128,),(128,256),(24,64),(64,)]) #Inception 4a 14x14x512
    x = Inception(x,[(112,),(144,288),(32,64),(64,)]) #Inception 4a 14x14x528
    x = Inception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 4a 14x14x832
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x) #7x7x832

    x = Inception(x,[(256,),(160,320),(32,128),(128,)]) #Inception 5a 7x7x832
    x = Inception(x,[(384,),(192,384),(48,128),(128,)]) #Inception 5b 7x7x1024

    #Using AveragePooling replace flatten
    x = AveragePooling2D(pool_size=(7,7),strides=(7,7),padding='same')(x)
    x =Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(1000,activation='relu')(x)
    x = Dense(classes,activation='softmax')(x)
    
    model=Model(input=inpt,output=x)
    
    return model

InceptionV1_model = InceptionV1(224,224,3,17)
InceptionV1_model.summary()

InceptionV1_model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),loss = 'categorical_crossentropy',metrics=['accuracy'])
History = InceptionV1_model.fit_generator(img_gen.flow(X_train*255, y_train, batch_size = 16),steps_per_epoch = len(X_train)/16, validation_data = (X_test,y_test), epochs = 30 )

#Plot Loss and accuracy
import matplotlib.pyplot as plt
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.show()

```



# 六、Inception V3

Christian 和他的团队都是非常高产的研究人员。2015 年 2 月，**Batch-normalized Inception** 被引入作为**Inception V2**。



2015年12月，他们发布了一个新版本的GoogLeNet([Inception V3](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1512.00567))模块和相应的架构，并且更好地解释了原来的GoogLeNet架构，GoogLeNet原始思想：



- 通过构建平衡深度和宽度的网络，最大化网络的信息流。在进入pooling层之前增加feature maps
- 当网络层数深度增加时，特征的数量或层的宽度也相对应地增加
- 在每一层使用宽度增加以增加下一层之前的特征的组合
- **只使用3x3卷积**

因此最后的模型就变成这样了：

​	![](c7.png)



# 七、ResNet

2015年12月[ResNet](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1512.03385v1.pdf)发表了，时间上大概与Inception v3网络一起发表的。

我们已经知道加深层对于提升性能很重要。但是，在深度学习中，过度加深层的话，会出现梯度消失、梯度爆炸、网络退化，导致最终性能不佳。 ResNet中，为了解决这类问题，导入了“快捷结构”（也称为“捷径”或“小路”）。导入这个快捷结构后，就可以随着层的加深而不断提高性能了（当然，层的加深也是有限度的）。 

![](09.png)

图，在连续2层的卷积层中，将输入x跳着连接至2层后的输出。这里的重点是，通过快捷结构，原来的2层卷积层的输出$F(x)$变成了$F(x) + x$。通过引入这种快捷结构，即使加深层，也能高效地学习。 

因为快捷结构只是原封不动地传递输入数据，所以反向传播时会将来自上游的梯度原封不动地传向下游。这里的重点是不对来自上游的度进行任何处理，将其原封不动地传向下游。因此，基于快捷结构，不用担心梯度会变小（或变大），能够向前一层传递“有意义的梯度”。通过这个快捷结构，之前因为加深层而导致的梯度变小的梯度消失问题就有望得到缓解。

![](c8.png)

ResNet通过以2个卷积层为间隔跳跃式地连接来加深层。另外，根据实验的结果，即便加深到150层以上，识别精度也会持续提高。并且，在ILSVRC大赛中， ResNet的错误识别率为3.5%（前5类中包含正确解这一精度下的错误识别率），令人称奇。 



```python

import numpy as np
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Dropout,BatchNormalization,AveragePooling2D,concatenate,Input, concatenate
from keras.models import Model,load_model
from keras.optimizers import Adam

#Load oxflower17 dataset
import tflearn.datasets.oxflower17 as oxflower17
from sklearn.model_selection import train_test_split
x, y = oxflower17.load_data(one_hot=True)

#Split train and test data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,shuffle = True)

#Data augumentation with Keras tools
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    )

#Define convolution with batchnromalization
def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = BatchNormalization(axis=3,name=bn_name)(x)
    return x
  
#Define Residual Block for ResNet34(2 convolution layers)
def Residual_Block(input_model,nb_filter,kernel_size,strides=(1,1), with_conv_shortcut =False):
    x = Conv2d_BN(input_model,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides,padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size,padding='same')
    
    #need convolution on shortcut for add different channel
    if with_conv_shortcut:
        shortcut = Conv2d_BN(input_model,nb_filter=nb_filter,strides=strides,kernel_size=kernel_size)
        x = add([x,shortcut])
        return x
    else:
        x = add([x,input_model])
        return x
    
#Built ResNet34
def ResNet34(width, height, depth, classes):
    
    Img = Input(shape=(width,height,depth))
    
    x = Conv2d_BN(Img,64,(7,7),strides=(2,2),padding='same')
    x = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')(x)  

    #Residual conv2_x ouput 56x56x64 
    x = Residual_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=64,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=64,kernel_size=(3,3))
    
    #Residual conv3_x ouput 28x28x128 
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)# need do convolution to add different channel
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=128,kernel_size=(3,3))
    
    #Residual conv4_x ouput 14x14x256
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)# need do convolution to add different channel
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=256,kernel_size=(3,3))
    
    #Residual conv5_x ouput 7x7x512
    x = Residual_Block(x,nb_filter=512,kernel_size=(3,3),strides=(2,2),with_conv_shortcut=True)
    x = Residual_Block(x,nb_filter=512,kernel_size=(3,3))
    x = Residual_Block(x,nb_filter=512,kernel_size=(3,3))


    #Using AveragePooling replace flatten
    x = GlobalAveragePooling2D()(x)
    x = Dense(classes,activation='softmax')(x)
    
    model=Model(input=Img,output=x)
    return model  

#Plot Loss and accuracy
import matplotlib.pyplot as plt
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.subplot(1,2,2)
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.show()
```





# 八、Inception v4 和 Inception-ResNet

2016年2月

Inception v4 和 Inception -ResNet 在同一篇论文[《Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning》](https://arxiv.org/abs/1602.07261).首先说明一下Inception v4**没有**使用残差学习的思想, 而出自同一篇论文的Inception-Resnet-v1和Inception-Resnet-v2才是Inception module与残差学习的结合产物。Inception-ResNet和Inception v4网络结构都是基于Inception v3的改进。

**Inception v4中的三个基本模块**：



<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v1.png"  width="180" height="240" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v2.png" width="180" height="240" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v3.png" width="180" height="240" ></div><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>











1. 左图是基本的Inception v2/v3模块，使用两个3x3卷积代替5x5卷积，并且使用average pooling，该模块主要处理尺寸为35x35的feature map；

2. 中图模块使用1xn和nx1卷积代替nxn卷积，同样使用average pooling，该模块主要处理尺寸为17x17的feature map；

3. 右图在原始的8x8处理模块上将3x3卷积用1x3卷积和3x1卷积。 

总的来说，Inception v4中基本的Inception module还是沿袭了Inception v2/v3的结构，只是结构看起来更加简洁统一，并且使用更多的Inception module，实验效果也更好。

下图左图为Inception v4的网络结构，右图为Inception v4的Stem模块：

<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v10.png"  width="250" height="350" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v11.png" width="250" height="350" ></div><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>




**Inception-Resnet-v1基本模块**：

<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v4.png"  width="180" height="240" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v5.png" width="180" height="240" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v6.png" width="180" height="240" ></div><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>















1. Inception module都是简化版，没有使用那么多的分支，因为identity部分（直接相连的线）本身包含丰富的特征信息；
2. Inception module每个分支都没有使用pooling；
3. 每个Inception module最后都使用了一个1x1的卷积（linear activation），作用是保证identity部分和Inception部分输出特征维度相同，这样才能保证两部分特征能够相加。

  

<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v12.png"  width="250" height="350" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v13.png" width="250" height="350" ></div><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>


















**Inception-Resnet-v2基本模块：**：

<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v7.png"  width="180" height="240" ></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="v8.png" width="180" height="240" ></div><div style="float:left;border:solid 1px 000;margin:2px;"><img src="v9.png" width="180" height="240" ></div><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>
Inception-Resnet-v2网络结构同Inception-Resnet-v1，Stem模块同Inception v4





# 九、Xception

2016年８月

[Xception](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1610.02357)是google继Inception后提出的对Inception v3的另一种改进，主要是采用depthwise separable convolution来替换原来Inception v3中的卷积操作。



**结构的变形过程如下**：

- 在 Inception 中，特征可以通过 1×1卷积，3×3卷积，5×5 卷积，pooling 等进行提取，Inception 结构将特征类型的选择留给网络自己训练，也就是将一个输入同时输给几种提取特征方式，然后做 concat 。Inception-v3的结构图如下:

  ![](x1.png)

- 对 Inception-v3 进行简化，去除 Inception-v3 中的 avg pool 后，输入的下一步操作就都是 1×1卷积：

  ![](x2.png)

- 提取 1×1卷积的公共部分：

  ![](x3.png)

- Xception（**极致的 Inception**）：先进行普通卷积操作，再对 1×1卷积后的每个channel分别进行 3×3卷积操作，最后将结果 concat：

  ![](x4.png)

**深度可分离卷积 Depthwise Separable Convolution**

传统卷积的实现过程：

![](x5.png)

Depthwise Separable Convolution 的实现过程：

![](x6.png)



**Depthwise Separable Convolution 与 极致的 Inception 区别：**

极致的 Inception：

​	第一步：普通 1×1卷积。

​	第二步：对 1×1卷积结果的每个 channel，分别进行 3×3卷积操作，并将结果 concat。

Depthwise Separable Convolution：

​	第一步：Depthwise 卷积，对输入的每个channel，分别进行 3×3 卷积操作，并将结果 concat。

​	第二步：Pointwise 卷积，对 Depthwise 卷积中的 concat 结果，进行 1×1卷积操作。

两种操作的循序不一致：Inception 先进行 1×1卷积，再进行 3×3卷积；Depthwise Separable Convolution 先进行 3×3卷积，再进行1×1 卷积。



# 十 、DenseNet

2016年8yue

[DenseNe](https://arxiv.org/abs/1608.06993)的基本思路与ResNet一致，但是它建立的是前面所有层与后面层的密集连接（dense connection），它的名称也是由此而来。DenseNet的另一大特色是通过特征在channel上的连接来实现特征重用（feature reuse）。这些特点让DenseNet在参数和计算成本更少的情形下实现比ResNet更优的性能，DenseNet也因此斩获CVPR 2017的最佳论文奖。

![](d1.png)



**设计理念**

相比ResNet，DenseNet提出了一个更激进的密集连接机制：即互相连接所有的层，具体来说就是**每个层都会接受其前面所有层作为其额外的输入**。ResNet是每个层与前面的某层（一般是2~3层）短路连接在一起，连接方式是通过**元素级**相加。而在DenseNet中，每个层都会与前面所有层在channel维度上连接（concat）在一起（这里各个层的特征图大小是相同的，后面会有说明），并作为下一层的输入。对于一个 $L$层的网络，DenseNet共包含 $\frac{L(L+1)}{2}$个连接，相比ResNet，这是一种密集连接。而且DenseNet是直接concat来自不同层的特征图，这可以实现特征重用，提升效率，这一特点是DenseNet与ResNet最主要的区别。



**网络结构**

CNN网络一般要经过Pooling或者stride>1的Conv来降低特征图的大小，而DenseNet的密集连接方式需要特征图大小保持一致。为了解决这个问题，DenseNet网络中使用DenseBlock+Transition的结构，其中DenseBlock是包含很多层的模块，每个层的特征图大小相同，层与层之间采用密集连接方式。而Transition模块是连接两个相邻的DenseBlock，并且通过Pooling使特征图大小降低。

![](d2.jpg)



在DenseBlock中，各个层的特征图大小一致，可以在channel维度上连接。DenseBlock中的非线性组合函数 $H$采用的是**BN+ReLU+3x3 Conv**的结构。另外值得注意的一点是，与ResNet不同，所有DenseBlock中各个层卷积之后均输出个**K**特征图，即得到的特征图的channel数为$k$，或者说采用$k$个卷积核。$k$ 在DenseNet称为growth rate，这是一个超参数。一般情况下使用较小的 $k$（比如12），就可以得到较佳的性能。假定输入层的特征图的channel数为 $k_0$，那么$l$层输入的channel数为$k_0 + k(l-1)$，因此随着层数增加，尽管**K**设定得较小，DenseBlock的输入会非常多，不过这是由于特征重用所造成的，每个层仅有 **K** 个特征是自己独有的。



![](d3.jpg)

由于后面层的输入会非常大，DenseBlock内部可以采用bottleneck层来减少计算量，主要是原有的结构中增加1x1 Conv，如图7所示，即**BN+ReLU+1x1 Conv+BN+ReLU+3x3 Conv**，称为DenseNet-B结构。其中1x1 Conv得到 $4k$个特征图它起到的作用是降低特征数量，从而提升计算效率。



对于Transition层，它主要是连接两个相邻的DenseBlock，并且降低特征图大小。Transition层包括一个1x1的卷积和2x2的AvgPooling，结构为**BN+ReLU+1x1 Conv+2x2 AvgPooling**。另外，Transition层可以起到压缩模型的作用。假定Transition的上接DenseBlock得到的特征图channels数为$m$，Transition层可以产生 个$[\theta m]$特征（通过卷积层），其中 $\theta \in (0, 1]$是压缩系数（compression rate）。当 $\theta =1$ 时，特征个数经过Transition层没有变化，即无压缩，而当压缩系数小于1时，这种结构称为DenseNet-C 。对于使用bottleneck层的DenseBlock结构和压缩系数小于1的Transition组合结构称为DenseNet-BC。






# 参考

​	https://www.cnblogs.com/CZiFan/p/9490565.html

​	https://www.zhihu.com/question/53727257/answer/136261195

​	https://blog.csdn.net/lk3030/article/details/84847879

​	https://blog.csdn.net/zzc15806/article/details/83504130

​	https://zhuanlan.zhihu.com/p/37189203



[https://medium.com/%E9%9B%9E%E9%9B%9E%E8%88%87%E5%85%94%E5%85%94%E7%9A%84%E5%B7%A5%E7%A8%8B%E4%B8%96%E7%95%8C/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-ml-note-cnn%E6%BC%94%E5%8C%96%E5%8F%B2-alexnet-vgg-inception-resnet-keras-coding-668f74879306](https://medium.com/雞雞與兔兔的工程世界/機器學習-ml-note-cnn演化史-alexnet-vgg-inception-resnet-keras-coding-668f74879306)

