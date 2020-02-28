#coding =utf-8
#导入需要的包

import tensorflow as tf
import getConfig

print('当前tensorflow版本号',tf.__version__)
print('当前的gpu是否可用',tf.test.is_gpu_available())
physical_device = tf.config.list_physical_devices('GPU')
print('当前可用的gpu:',len(physical_device))


gConfig={}
gConfig = getConfig.get_config(config_file='config.ini')


#定义cnnModel方法类，object类型，这样在执行器中可以直接实例化一个CNN进行训练
class cnnModel(object):

    def __init__(self,rate):

        #定义神经元失效的概率
        self.rate=rate

    #定义一个网络模型，这是使用tf.keras.Sequential进行网络模型定义的标准形式
    def createModel(self):
        #实例化一个Sequential.接下来就可以使用add方法来叠加所需要的网络层
        model = tf.keras.Sequential()

        '''添加一个二维卷积层，输出数据维度为32，卷积核维度为 3 x 3 。 输入数据维度为【32，32，3】。 这里的
        维度是WHC格式的，意思是输入图像像素为32 x 32 的尺寸，使用3通道也就是RGB的像素值。同样，如果图像时64 x 64尺寸的，
        则可以设置输入数据维度为【64，64，3】，如果图像尺寸不统一，则要进行尺寸转化处理
        '''
        model.add(tf.keras.layers.Conv2D(32,(3,3),kernel_initializer = 'he_normal', strides = 1, padding = 'same', activation = 'relu',input_shape = [32,32,3],name = 'conv1'))

        #添加一个二维池化层，使用最大值池化，池化维度2 x 2。 也就是说，在一个2 x 2 的像素区域内去一个像素最大值作为该区域的像素特征


        model.add(tf.keras.layers.MaxPool2D((2,2),strides = 1, padding = 'same',name='pool1')) #一般池化层是不用padding 的呀？

        #添加一个批量池化层 Batch Normalization ： 这里还没学会，要去搞懂它
        model.add(tf.keras.layers.BatchNormalization())

        #添加第二个二维卷积层，输出数据维度为64，卷积核维度是3x3
        model.add(tf.keras.layers.Conv2D(64,(3,3),kernel_initializer='he_normal',strides = 1, padding = 'same', activation='relu',name='conv2'))

        #添加第二个二维池化层，使用最大值池化，池化维度2x2
        model.add(tf.keras.layers.MaxPool2D((2,2),strides = 1,padding ='same',name='pool2'))

        #添加一个批量池化层，BatchNormalization
        model.add(tf.keras.layers.BatchNormalization())

        #添加第三个卷积层，输出数据维度为128，卷积和维度是3 x 3
        model.add(tf.keras.layers.Conv2D(128,(3,3),kernel_initializer = 'he_normal',strides = 1, padding = 'same', activation ='relu', name='conv3'))

        #添加第三个二维池化层，使用最大值池化，池化维度为2 x 2
        model.add(tf.keras.layers.MaxPool2D((2,2),strides = 1,padding ='same',name='pool3'))

        #添加一个批量池化层 BatchNormalization
        model.add(tf.keras.layers.BatchNormalization())

        '''在经过卷积和池化完成特征提取之后，紧接着是一个全连接的深度神经网络。在将数据输入深度神经网络之前主要进行数据的Flatten操作。
        就是将之前的长 ， 宽，像素值三个维度的数据压平成一个维度。这样可以减少参数的数量。因此，在卷积层和全链接神经网络之间添加一个Flatten层。
        
        '''
        model.add(tf.keras.layers.Flatten(name='flatten'))

        #添加一个Dropout 层，防止过拟合，加快训练速度
        model.add(tf.keras.layers.Dropout(rate=self.rate,name='d3'))

        ''''最后一层作为输出层，因为是进行图像的10分类，所以输出的数据维度是10，使用softmax作为激活函数，softmax是一个在多分类问题上使用的激活函数
        如果是二分类问题，则softmax 和sigmoid的作用是类似的'''
        model.add(tf.keras.layers.Dense(10,activation='softmax'))


        '''在完成神经网络的设计后，我们需要对网络模型进行编译，生成可以训练的模型，在进行编译时，需要定义损失函数loss和优化器optimizer
        模型评价标准（metrics），这些都可以使用高阶API直接调用
        '''
        model.compile(loss='categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(),metrics = ['accuracy'])

        return model