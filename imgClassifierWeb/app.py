import flask
import werkzeug
import os
import execute
import getConfig
import requests
import pickle
from flask import request,jsonify
import numpy as np
from PIL import Image
gConfig = {}
gConfig = getConfig.get_config()


#实例化一个flask应用，命名为imgClassifierWeb
app= flask.Flask('imgClassifierWeb')

#定义预测函数
def CNN_predict():
    #获取图片分类名称存放路径
    file = getConfig['dataset_path']+'batches.meta'

    #读取图片分类名称，并保存到一个字典中
    patch_bin_file = open(file,'rb')

    label_names_dict = pickle.load(patch_bin_file)['label_names']

    #全局声明一个文件名
    global secure_filename

    #从本地目录中读取需要分类的图片
    img = Image.open(os.path.join(app.root_path,secure_filenname))

    #将读取的像素格式转换为RGB,并分别获取RGB通道对应的像素数据
    r,g,b = img.split()

    #分别将获取的像素数据放入数组中
    r_arr = np.array(r)
    g_arr = np.array(g)
    b_arr = np.array(b)

    #将三个数组进行拼接
    img = np.concatenate((r_arr,g_arr,b_arr))

    #对拼接狗的数据进行维度变换和归一化处理
    image = img.reshape([1,32,32,3])/255

    #调用执行器execute的predict函数对图像数据进行预测
    predicted_class = execute.predict(image)

    #将返回结果用页面模版渲染出来
    return flask.render_template(template_name_or_list = 'prediction_result.html', predicted_class = predicted_class)

app.add_url_rule(rule='/predict/',endpoint = 'predict',view_func = CNN_predict)

def upload_image():
    global secure_filename
    if flask.request.method =='POST': #设置request的模式为POST

        #获取需要分类的图片
        img_file = flask.request.files['image_file']

        #生成一个没有乱码的文件名
        secure_filename = werkzeug.secure_filename(img_file,filename)

        #获取图片的保存路径
        img_path = os.path.join(app.root_path,secure_filename)

        #将图片保存在应用的根目录下
        img_fille.save(img_path)

        print('图片上传成功')

        return flask.redirect(flask.url_for(endpoint='predict'))

    return "图片上传失败"

#增加图拍呢上传的路由入口
app.add_url_rule(rule='/upload/',endpoint='upload',view_func=upload_image,methods=['POST'])

def redirect_upload():
    return flask.render_template(template_name_or_list=  'upload_image.html')


#增加默认主页的路由入口
app.add_url_rule(rule='/',endpoint='homepage',view_func=redirect_upload)


if __name__ == '__main__':

    app.run(host='0.0.0.0',port = 7777,debug = False)
