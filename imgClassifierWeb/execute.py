import tensorflow as tf
import numpy as np
from cnnModel import cnnModel
import os
import pickle
import time
import getConfig
import sys
import random

gConfig = {}
#调用get_config读取配置文件中的参数

gConfig = getConfig.get_config(config_file='config.ini')

#定义数据读取函数，在这个函数中完成数据读取，格式转换的操作
def read_data(dataset_path,im_dim,num_channels,num_files,images_per_file):
    #获取文件夹中的数据文件名

    files_names = os.listdir(dataset_path)

    #获取训练集中训练文件的名称

    '''在CIFAR-10中已经为我们标注和准备好了数据，如果一时找不到合适的高质量的标注训练集，那么就使用CIFAR-10作为训练集
    训练集中一共有50，000个样本，放到5个二进制文件中，每个样本有3072个像素点，维度是32 x 32 x 3
    '''
    #创建空的多维数组用于存放图像二进制数据
    dataset_array = np.zeros(shape = (num_files*images_per_file, im_dim,im_dim,num_channels))
    #创建空的数组用于存放图像的标注悉尼下
    dataset_labels = np.zeros(shape = (num_files*images_per_file), dtype=np.uint8)


    index = 0
    #从训练集中读取二进制数据并将其维度转换成32*32*3

    for file_name in files_names:
        if file_name[0:len(file_name)-1] == 'data_batch_':
            print('正在处理数据：', file_name)
            data_dict = unpickle_patch(dataset_path+file_name)
            images_data = data_dict[b"data"]
            print(images_data.shape)
            #将格式转换为32 x 32 x 3 形状
            images_data_reshaped = np.reshape(images_data,newshape = (len(images_data),im_dim,im_dim,num_channels))

            #将维度转换后的图像数据存入指定数组内
            dataset_array[index * images_per_file:(index+1)*images_per_file,:,:,:] = images_data_reshaped

            #将维度转换后的标注数据存入指定的数组内
            dataset_labels[index*images_per_file:(index+1)*images_per_file] = data_dict[b"labels"]

            index = index +1
    return dataset_array, dataset_labels


def unpickle_patch(file):
    #打开文件，读取二进制文件，返回读取到的数据
    patch_bin_file = open(file,'rb')
    patch_dict = pickle.load(patch_bin_file,encoding='bytes')

    return patch_dict



#定义模型实例化函数，主要判断是否有预训练模型，如果有，则优先加载预训练模型，判断是否有已经保存的训练文件
#如果有则加载文件继续训练，否则构建实例化神经网络模型进行训练

def create_model():
    #判断是否存在预训练模型
    if 'pretrained_model' in gConfig:
        model = tf.keras.models.load_model(gConfig['pretrained_model'])
        return model

    ckpt = tf.io.gfile.listdir(gConfig['working_directory'])

    #判断是否存在模型文件，如果存在则加载模型继续训练，如果不存在，则新建模型相关文件
    if ckpt:
        model_file = os.path.join(gConfig['working_directory'],ckpt[-1])
        print('Reading model parameters from %s'%model_file)
        model = tf.keras.models.load_model(model_file)
        return model
    else:
        model = cnnModel(gConfig['keeps'])
        model = model.createModel()
        return model


#读取训练集的数据，根据read_data函数的参数定义需要传入dataset_path,im_dim,num_channels,num_files,images_per_file
dataset_array,dataset_labels= read_data(dataset_path=gConfig['dataset_path'],
                                        im_dim=gConfig['im_dim'],
                                        num_channels=gConfig['num_channels'],
                                        num_files=gConfig['num_files'],
                                        images_per_file=gConfig['images_per_file'])


test_array,test_labels = read_data(dataset_path=gConfig['test_path'],
                                        im_dim=gConfig['im_dim'],
                                        num_channels=gConfig['num_channels'],
                                        num_files=1,
                                        images_per_file=gConfig['images_per_file'])

#对训练输入数据进行归一化处理，取值范围为（0，1）
dataset_array = dataset_array.astype('float32')/255
test_array = test_array.astype('float32')/255


#对标注数据进行one-hot编码
dataset_labels = tf.keras.utils.to_categorical(dataset_labels,10)
test_labels = tf.keras.utils.to_categorical(test_labels,10)

#定义训练函数
def train():
    #实例化一个神经网络模型
    model = create_model()
    print(model.summary())

    #开始进行模型训练
    model.fit(dataset_array,dataset_labels,epochs = gConfig['epochs'],validation_data = (test_array,test_labels),verbose = 1)

    #将完成训练的模型保存起来
    filename= 'cnn_model.h5'

    checkpoint_path = os.path.join(gConfig['working_directory'],filename)

    model.save(checkpoint_path)

#
#定义预测函数，加载所保存的模型文件并进行预测
def predict(data):
    file = gConfig['dataset_path']+'batches.meta'

    patch_bin_file= open(file,'rb')
    label_names_dict = pickle.load(patch_bin_file)['label_names']

    #获取最新的模型文件路径
    model= create_model()


    #对数据进行预测
    prediction=model.predict(data)

    #使用argmax获取预测结果
    index =tf.math.argmax(prediction[0]).numpy()

    #返回预测的分类名称
    return label_names_dict[index]


if __name__ == '__main__':
    if gConfig['mode']=='train':
        train()
    elif gConfig['mode']=='serve':
        print('请使用：python3 app.py')