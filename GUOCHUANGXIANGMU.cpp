 //cnnģ�� 
 #����ģ�ͽ���Ԥ�⣺
from skimage import io,transform
import tensorflow as tf
import numpy as np


# ģ�ͱ����ַ ����Ϊ�Լ��ģ�
model_path = 'D:/DLLab/Model/'
# ģ�ͱ�������
model_name = 'CNN_model'

flower_dict = {0:'Բ������',1:'��',2:'���������ӣ�1��',3:'���������ӣ�2��'}

#w=100
#h=100
#c=3

img=io.imread(r'five.JPG')
img=transform.resize(img,(100,100))
data=[]
data1=np.asarray(img)
data.append(data1)
"""
#��ȡͼ�����ŵ�����ߴ�
def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

# ׼����������
data = []
data1 = read_one_image('first.JPG')
data.append(data1)
"""


# �����Ự
with tf.Session() as sess:
    #������µı�����
    model_file=tf.train.latest_checkpoint(model_path)
    # ��ģ����import����ͼ
    saver = tf.train.import_meta_graph(model_file+'.meta')
    # ��ģ����restoreȨ����Ϣ
    saver.restore(sess, model_file) 

    graph = tf.get_default_graph()
    
    # ����ͼ���ҵ�����ڵ�
    x = graph.get_tensor_by_name("x:0")
    
    # �����feed��Ϣ
    feed_dict = {x:data}

    # ����ͼ���ҵ�����ڵ�
    logits = graph.get_tensor_by_name("logits_eval:0")

    # run�ػ� �ṩ���� ������
    classification_result = sess.run(logits,feed_dict)

    #����numpy������ʾ4λС��
    np.set_printoptions(precision=4, suppress=True)
    #��ӡ��Ԥ�����
    print(classification_result)
    #��ӡ��Ԥ�����ÿһ�����ֵ������
    print(tf.argmax(classification_result,1).eval())
    #��������ͨ���ֵ��Ӧ���ķ���
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("��",i+1,"������Ԥ��:"+flower_dict[output[i]])
        plate=output[i]
        print(plate)

//ģ��ѵ�� 
# -*- coding:uft-8

from  skimage import io,transform
import glob
import os
import tensorflow as tf
import numpy as np
import time

# ���ݼ��ĵ�ַ  ��Ϊ���Լ���
path = 'C:/Users/24223/Desktop/����/'
# ģ�ͱ����ַ
model_path = 'D:/DLLab/Model/'
# ģ�ͱ�������
model_name = 'CNN_model'
# tensorboard dir
tb_dir = 'D:/DLLab/tbdir/'

# ����ͼƬ��СΪ100*100
w = 100
h = 100
c = 3

#��ȡͼƬ
def read_img(path):
    cate=[path+x for x in os.listdir(path) if os.path.isdir(path+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
data,label=read_img(path)

#����˳��
num_example=data.shape[0] 
arr=np.arange(num_example)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]


#���������ݷ�Ϊѵ��������֤��
ratio=0.8
s=np.int(num_example*ratio)
x_train=data[:s]
y_train=label[:s]
x_val=data[s:]
y_val=label[s:]

print(len(x_train))
print(len(x_val))

#-----------------��������----------------------
#ռλ��
x=tf.placeholder(tf.float32,shape=[None,w,h,c],name='x')
y_=tf.placeholder(tf.int32,shape=[None,],name='y_')

def inference(input_tensor, train, regularizer):
    #with tf.variable_scope('conv'):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")

    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer5-conv3"):
        conv3_weights = tf.get_variable("weight",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))

    with tf.name_scope("layer6-pool3"):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.variable_scope("layer7-conv4"):
        conv4_weights = tf.get_variable("weight",[3,3,128,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
        conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

    with tf.name_scope("layer8-pool4"):
        pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        nodes = 6*6*128
        reshaped = tf.reshape(pool4,[-1,nodes])
    #with tf.variable_scope('fc'):
    with tf.variable_scope('layer9-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer10-fc2'):
        fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
        if train: fc2 = tf.nn.dropout(fc2, 0.5)

    with tf.variable_scope('layer11-fc3'):
        fc3_weights = tf.get_variable("weight", [512, 5],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.get_variable("bias", [5], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc2, fc3_weights) + fc3_biases

    return logit


regularizer = tf.contrib.layers.l2_regularizer(0.0001)
logits = inference(x,False,regularizer)
logits_eval = tf.nn.softmax(logits, name='logits_eval') 

#---------------------------�������---------------------------
# �������ͼ��tensorboard�鿴

sess=tf.Session()  
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter( tb_dir, sess.graph)

#��ʧ����
loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)

loss_reg = tf.add_n(tf.get_collection('losses'))
loss = loss + loss_reg

#�Ż���
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

writer = tf.summary.FileWriter( tb_dir, sess.graph)

#��ȷ�ĸ���
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
#��ȷ��
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#����һ��������������ȡ����
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

#ѵ���Ͳ������ݣ�n_epoch��ѵ��������

n_epoch=5                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
batch_size=3  #���δ�С
display_step = 10
saver=tf.train.Saver()
sess=tf.Session()  
sess.run(tf.global_variables_initializer())

# �������ͼ��tensorboard�鿴
writer = tf.summary.FileWriter(tb_dir, sess.graph)

for epoch in range(n_epoch):
    start_time = time.time()

    #training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _,err,ac=sess.run([train_op,loss,acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err; train_acc += ac; n_batch += 1
        if n_batch % display_step == 1:
            print("  batch acc: %f      batch loss: %f" % (np.sum(ac),np.sum(err)))
    print("   train loss: %f" % (np.sum(train_loss)/ n_batch))
    print("   train acc: %f" % (np.sum(train_acc)/ n_batch))
    print("  n_batch:%f"%(n_batch))

    #validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
        err, ac = sess.run([loss,acc], feed_dict={x: x_val_a, y_: y_val_a})
        val_loss += err; val_acc += ac; n_batch += 1
    print("   validation loss: %f" % (np.sum(val_loss)/ n_batch))
    print("   validation acc: %f" % (np.sum(val_acc)/ n_batch))
    print("  n_batch:%f"%(n_batch))
    print('-------------------------------------------------------')
    saver.save(sess, model_path+model_name, global_step=epoch)
#saver.save(sess,model_path+model_name)
sess.close()

model_file=tf.train.latest_checkpoint(model_path)
print(model_file)

//��Ŀ������� 
import cv2
import time

def get_photo():
    cap=cv2.VideoCapture(0)
    ret,frame=cap.read()
    if ret==True:
        print("true")
    cv2.imwrite('first.JPG',frame)

    cap.release()
    cv2.destroyAllWindows()
    print("photo is ok!")

from skimage import io,transform
import tensorflow as tf
import numpy as np

import requests
import random
import json
import time
import re

import serial

#����api����Կ
API_KEY = 'j5XSXv=pHbyD=P=MWg3kbuHlzso='
#�������ݵ�url
URL_datapoints= "http://api.heclouds.com/devices/522843297/datapoints"
#����ͼƬ��url
URL_picture = "http://api.heclouds.com/bindata"
#��ȡ�����url
URL_command = "http://api.heclouds.com/devices/522843297/datastreams/"
#ÿ������ʱ�䷢��һ��
interval_time = 10
#��ǰʱ��
now_time = int(time.time())
#��Ҫ�ϴ���ͼƬ·��
#path_image = ["E:/python/onenet/1.jpg", "E:/python/onenet/2.jpg", "E:/python/onenet/3.jpg","E:/python/onenet/4.jpg"]
path_image = "C://Users/24223/first.jpg"

#ʶ������

def Cnn_plate():
    model_path = 'D:/DLLab/Model/'
    model_name = 'CNN_model'
    flower_dict = {0:'Բ������',1:'��',2:'����������'}
    with tf.Session() as sess:
        model_file=tf.train.latest_checkpoint(model_path)
        saver = tf.train.import_meta_graph(model_file+'.meta')
        saver.restore(sess, model_file) 

        graph = tf.get_default_graph()

        
        get_photo()
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        img=io.imread(r'first.JPG')
        img=transform.resize(img,(100,100))
        data=[]
        data1=np.asarray(img)
        data.append(data1)

        x = graph.get_tensor_by_name("x:0")

        feed_dict = {x:data}

        logits = graph.get_tensor_by_name("logits_eval:0")

        classification_result = sess.run(logits,feed_dict)

        np.set_printoptions(precision=4, suppress=True)
        print(classification_result)
        print(tf.argmax(classification_result,1).eval())
        output = []
        output = tf.argmax(classification_result,1).eval()

        for i in range(len(output)):
            print("����Ԥ��:"+flower_dict[output[i]])
            plate=output[i]
            print(plate)
    print(type(plate))
    return int(plate)
    
def picture_post():
    #����ͷ
    headers = {
        'api-key': API_KEY,
        "Content-Type": "image/jpg",
    }
    #�������
    params = {
        'device_id': 522843297,
        'datastream_id': "image"
    }

    #for i in range(4):
    with open(path_image, 'rb') as f:
         requests.post(url=URL_picture, headers=headers, params=params , data=f)
    print("succ")

def datapoints_post():
    #��ǰʱ��
    #now_time = time
    #����ת��
    machine_speed = 0
    #ʶ���plate
    which_plate = Cnn_plate()


    headers = {
        'api-key': API_KEY,
    }
    params = {
        'device_id': 522843297,
    }
    #���ݵ�ֵ
    values = {
        "datastreams": [
            # ������������
            {
            "id":"speed",
            "datapoints":[{
                #��ǰ�����ת��
                "value": machine_speed
                
            }],
            },

            {
            "id": "plate",
            "datapoints": [{
                # �������ĸ�
                 "value": which_plate
                
            }
            ],
        }
        ]
    }
    jdata = json.dumps(values)
    requests.post(url=URL_datapoints, headers=headers, data=jdata)


#��ȡָ��
def Get_command():
    headers={
        "api-key":API_KEY
    }
    #��ȡ��������������command
    URL_command_command = URL_command+"command"
    #��ȡ����
    command = requests.get(url=URL_command_command, headers=headers)
    #��ȡָ���ֵ
    command_value =re.findall(r"current_value.*\d", command.text)[0].split(':')[1]
    #����Ƿ��ȡ����ֵ
    #print(command.text)
    #��ȡ����ֵ
    #print(command_value)
    return command_value


"""
if __name__ == "__main__":

    # ��һ��ָ��
    last_command = Get_command()

    while(1):

        # ��ǰʱ��
        now_time_circle = int(time.time())

        #ÿ��3s��ȡһ��ָ��
        if( (now_time_circle-now_time)%interval_time==0 ):
            # ��������
            datapoints_post()
            picture_post()
            #��ȡ��ָ��
            #��Ϣ1s
            #time.sleep(1)
            #��һ��ָ��
            now_command=Get_command()
            #�������ָ����ͬ������
            if(now_command!=last_command):
                print("ʱ��", time.strftime("%S",time.localtime(now_time_circle)),":",now_command)
"""

            
if __name__ == '__main__':
    serial=serial.Serial('COM6',9600,timeout=0.5)
    if serial.isOpen():
        print("open success")
    else:
        print("open failes")

# ��һ��ָ��
last_command = Get_command()

try:
    while True:
        count=serial.inWaiting()
        if count>0:
            data=serial.read(count)
            print(data)
            if data != b' ':
                print("receive: ",data)
                # ��������
                datapoints_post()
                picture_post()
                #��ȡ��ָ��
                #��Ϣ1s
                #time.sleep(1)
                #��һ��ָ��
                now_command=Get_command()
                #�������ָ����ͬ������
                if(now_command!=last_command):
                    print("ʱ��", time.strftime("%S",time.localtime(now_time_circle)),":",now_command)
except KeyboardInterrupt:
    if serial !=None:
        serial.close()

//������ҳ���� 

/*
				*{
						padding: 0px;
						margin: 0px;
						list-style:none;
						background-color: rgba(250,250,250,1);
					}
					.nav{
						list-style: none;
						width: 100%;
						/*height: 45px;*/
						overflow: hidden;
						zoom: 1;
					}
					.nav li{
						float: left;
						width: 25%;
					}
					.nav a{
						width: 100%;
						display: inline-block;
						text-align: center;
						padding: 5px 0px;
						text-decoration: none;
						color: black;
						font-weight: bold;
					}
					.nav a:hover{
						background-color: black;
						color: white;
					}
					.logo {
						width: 30%;
						height: 100px;
						margin-right: 50px;
						margin-left: 20px;
						margin-top: 20px;
						float: left;
					}
					.content{
						position: absolute;
						width: 100%;
						top: auto;
					}
					.right{
						position: absolute;
						top: 40px;
						right: 0;
						left: 32%;
						float: left;
					}
					.wrap{
						height:400px;
						width: 90%;
						margin:auto;
						overflow: hidden;
						position: relative;
						margin:auto;
					}
					.wrap ul{
						position:absolute;
					} 
					.wrap ul li{
						height:400px;
					}
					.wrap ol{
						position:absolute;
						right:47%;
						bottom:10px;
						background: none;
					}
					.wrap ol li{
						width: 20px;
						height: 20px;
						border-radius: 20px;
						border:solid 1px #666;
						margin-left:10px;
						float:left;
						line-height:center;
						text-align:center;
						cursor:pointer;
					}
					.wrap ol .on{
						background:mediumblue;
						color:#fff;
					}
					.roundedRectangle1{
						height: 300px;
						width: 700px;
						background: dodgerblue;
						box-shadow: 5px 5px 5px #666666;
						border-radius: 15px;
						margin-left: 30%;
					}
					.roundedRectangle1 p{
						font-family: ����;
						font-size: 25px;
						color:black;
						margin-right: 20px;
						margin-left: 20px;
						background: dodgerblue;
						line-height:1.5;
					}
					.roundedRectangle2{
						height: 400px;
						width: 600px;
						background: dodgerblue;
						border-radius: 15px;
						margin-left:10%;
						margin-top: 100px;
						box-shadow: 5px 5px 5px #666666;
					}
					.bottom{
						font-size: 20px;
						margin-top: 70px;
						font-family: ����;
						line-height:3;
						height: 350px;
						background: #000000;
						color: white;
					}
					.bottom_left{
						width: 50%;
						float: left;
						background: #000000;
					}
					.bottom_left p{
						margin-left: 20%;
						background: #000000;
					}
					.bottom_right{
						width: 50%;
						float: right;
						background: #000000;
					}
					.bottom_right p{
						margin-left: 15%;
						background: #000000;
					}
					

//js
window.onload=function(){
		var wrap=document.getElementById('wrap'),
		pic=document.getElementById('pic').getElementsByTagName("li"),
		list=document.getElementById('list').getElementsByTagName('li'),
		index=0,
		timer=null;
		
		// ���岢�����Զ����ź���
		timer = setInterval(autoPlay, 2000);
		
		// ��껮����������ʱֹͣ�Զ�����
		wrap.onmouseover = function () {
		clearInterval(timer);
		}
		
		// ����뿪��������ʱ������������һ��
		wrap.onmouseout = function () {
		timer = setInterval(autoPlay, 2000);
		}
		// �����������ֵ���ʵ�ֻ����л�����Ӧ��ͼƬ
		for (var i = 0; i < list.length; i++) {
		list[i].onmouseover = function () {
		clearInterval(timer);
		index = this.innerText - 1;
		changePic(index);
		};
		};
		
		function autoPlay () {
		if (++index >= pic.length) index = 0;
		changePic(index);
		}
		
		// ����ͼƬ�л�����
		function changePic (curIndex) {
		for (var i = 0; i < pic.length; ++i) {
		pic[i].style.display = "none";
		list[i].className = "";
		}
		pic[curIndex].style.display = "block";
		list[curIndex].className = "on";
		}
		
		};
		
//html
<!DOCTYPE html>
<html>
	<head>
		<meta charset="utf-8">
		<title>���ܲ;߷ּ�ϵͳ��˾����</title>
		<link rel="stylesheet" type="text/css" href="css/new_index.css"/>
		<script src="js/lunbo.js" type="text/javascript"></script>
	</head>
	<body>
		<div class="logo">
			<img src="img/name.png"alt="����"/>
		</div>
		<div class="right">
			<ul class="nav">
				<li><a href="#sy">��ҳ</a></li>
				<li><a href="#jj">���ܲ;߷ּ�ϵͳ���</a></li>
				<li><a href="#cgzs">�ɹ�չʾ</a></li>
				<li><a href="#lxwm">��ϵ����</a></li>
		</ul>
		<HR align=right color='black' SIZE=0.5>
		</div>
		<div id="sy" style="margin-bottom: 60px;">
			<div class="wrap" id='wrap'>
			<ul id="pic">
				<li><img src="img/csd.png" style="width: 1200px; height: 400px;"></li>
				<li><img src="img/cgq.png" style="width: 1200px; height: 400px;"></li>
				<li><img src="img/jxb.png" style="width: 1200px;height: 400px;"></li>
				<li><img src="img/sxt.png" style="width: 1200px;height: 400px;"></li>
			</ul>
			<ol id="list">
			<li class="on">1</li>
			<li>2</li>
			<li>3</li>
			<li>4</li>
			</ol>
			</div>
		</div>
		<div id="jj" class="roundedRectangle1">
			<p style="margin-top: 20px;">��������</p>
			<p style="font-size: 20px; margin-left: 40px;">�;����ּܷ�ϵͳ�ǲ�����ҵ�豸�о��и����ܡ���Ч�ʡ������򵥷�����˻��Ѻõ�һ���Զ�����еϵͳ��
			������������ˮƽ����ߣ����������˹���һ�����ƣ����Ҳ;����ּܷ�ϵͳ���зǳ��������г���
			������ȫ�������У�����Ϳ��������ȾͲͺ�;߻��ա���ͳ�òͳ����;߻��յ��������˹����Ͷ�ǿ�ȴ�
			Ч�ʵ��£�ʱЧ�Բ�ر��ǾͲ͸߷��ڣ�����յĲ;������������Լ�ʱ���;߻��գ����²;ߴ����ѻ���
			���豸���ܺõĸı�˲������档�����ۿ��ٽ����̺������Զ����룬�ﵽ��Ч��ʡ��ʡʱ���ɾ�������
			Ӫ�������Ͳͻ�����Ŀ�ġ�</p>
		</div>
		<div id="cgzs" class="roundedRectangle2">
			<video width="500" height="300" style="margin-left: 50px;margin-top: 50px;" controls autoplay poster="img/firstpage.png">
				<source src="img/video.mp4" type="video/mp4">
				<!--��Ϊ�ҵ���Ƶ��ʽ��ogg������·������Ƶ��ʽΪogg���±�ע�͵ĵط������������Ƶʲô��ʽ��Ūʲô��ʽ��-->
			    <!--<source src="img/video/movie.mp4" type="video/mp4">
			    <source src="movie.webm" type="video/webm">-->
			</video>

		</div>
		<div>
			<div id="lxwm" class="bottom">
				<div class="bottom_left">
					<p style="margin-top: 50px;">�ͷ��绰��0510-83733088</p>
					<p>����ʱ�䣺��һ����ĩ9:00 �� 18:30</p>
				</div>
				<div class="bottom_right">
					<p style="margin-top: 50px;">��ϵ�绰:13338760288</p>
					<p style="width: 60%;">��˾��ַ������ʡ�人�к�ɽ������·152�Ż���ʦ����ѧ</p>
				</div>
			</div>
		</div>
	</body>
</html>
					
					


*/ 

//��½ҳ��
/*

//css
@charset "UTF-8";
.animated {
  -webkit-animation-duration: 1s;
  animation-duration: 1s;
  -webkit-animation-fill-mode: both;
  animation-fill-mode: both;
}

.animated-fast {
  -webkit-animation-duration: .5s;
  animation-duration: .5s;
  -webkit-animation-fill-mode: both;
  animation-fill-mode: both;
}

@-webkit-keyframes fadeIn {
  from {
    opacity: 0;
    -ms-transform: scale(0.95);
    -webkit-transform: scale(0.95);
    transform: scale(0.95);
  }

  to {
    opacity: 1;
    -ms-transform: scale(1.0);
    -webkit-transform: scale(1.0);
    transform: scale(1.0);
  }
}

@keyframes fadeIn {
  from {
    opacity: 0;
    -ms-transform: scale(0.95);
    -webkit-transform: scale(0.95);
    transform: scale(0.95);
  }

  to {
    opacity: 1;
    -ms-transform: scale(1.0);
    -webkit-transform: scale(1.0);
    transform: scale(1.0);
  }
}

.fadeIn {
  -webkit-animation-name: fadeIn;
  animation-name: fadeIn;
}

body {
  font-family: "Open Sans", Arial, sans-serif;
  line-height: 1.5;
  font-size: 16px;
  color: #848484;
  background-color: #f0f0f0;
  background-repeat: no-repeat;
  background-image: url(../images/logo.png);
  background-size:100% 100%; 
  background-attachment: fixed;
}

a {
  color: #33cccc;
  -moz-transition: all 0.3s ease;
  -o-transition: all 0.3s ease;
  -webkit-transition: all 0.3s ease;
  -ms-transition: all 0.3s ease;
  transition: all 0.3s ease;
}
a:hover {
  color: #29a3a3;
}

.menu {
  padding: 0;
  margin: 30px 0 0 0;
}
.menu li {
  list-style: none;
  margin-bottom: 10px;
  display: -moz-inline-stack;
  display: inline-block;
  zoom: 1;
  *display: inline;
}
.menu li a {
  padding: 5px;
}
.menu li.active a {
  color: #b3b3b3;
}

.fh5co-form {
  padding: 30px;
  margin-top: 4em;
  -webkit-box-shadow: -4px 7px 46px 2px rgba(0, 0, 0, 0.1);
  -moz-box-shadow: -4px 7px 46px 2px rgba(0, 0, 0, 0.1);
  -o-box-shadow: -4px 7px 46px 2px rgba(0, 0, 0, 0.1);
  box-shadow: -4px 7px 46px 2px rgba(0, 0, 0, 0.1);
  background: #ffffff;
}
.style-2 .fh5co-form {
  -webkit-box-shadow: -4px 7px 46px 2px rgba(0, 0, 0, 0.1);
  -moz-box-shadow: -4px 7px 46px 2px rgba(0, 0, 0, 0.1);
  -o-box-shadow: -4px 7px 46px 2px rgba(0, 0, 0, 0.1);
  box-shadow: -4px 7px 46px 2px rgba(0, 0, 0, 0.1);
}
@media screen and (max-width: 768px) {
  .fh5co-form {
    padding: 15px;
  }
}
.fh5co-form h2 {
  text-transform: uppercase;
  letter-spacing: 2px;
  font-size: 20px;
  margin: 0 0 30px 0;
  color: #000000;
}
.fh5co-form .form-group {
  margin-bottom: 30px;
}
.fh5co-form .form-group p {
  font-size: 14px;
  color: #9f9f9f;
  font-weight: 300;
}
.fh5co-form .form-group p a {
  color: #000000;
}
.fh5co-form label {
  font-weight: 300;
  font-size: 14px;
  font-weight: 300;
}
.fh5co-form .form-control {
  font-size: 16px;
  font-weight: 300;
  height: 50px;
  padding-left: 0;
  padding-right: 0;
  border: none;
  border-bottom: 1px solid rgba(0, 0, 0, 0.1);
  -webkit-box-shadow: none;
  -moz-box-shadow: none;
  -o-box-shadow: none;
  box-shadow: none;
  -webkit-border-radius: 0px;
  -moz-border-radius: 0px;
  -ms-border-radius: 0px;
  border-radius: 0px;
  -moz-transition: all 0.3s ease;
  -o-transition: all 0.3s ease;
  -webkit-transition: all 0.3s ease;
  -ms-transition: all 0.3s ease;
  transition: all 0.3s ease;
}
.fh5co-form .form-control::-webkit-input-placeholder {
  color: rgba(0, 0, 0, 0.3);
  text-transform: uppercase;
}
.fh5co-form .form-control::-moz-placeholder {
  color: rgba(0, 0, 0, 0.3);
  text-transform: uppercase;
}
.fh5co-form .form-control:-ms-input-placeholder {
  color: rgba(0, 0, 0, 0.3);
  text-transform: uppercase;
}
.copyrights{
	text-indent:-9999px;
	height:0;
	line-height:0;
	font-size:0;
	overflow:hidden;
}
.fh5co-form .form-control:-moz-placeholder {
  color: rgba(0, 0, 0, 0.3);
  text-transform: uppercase;
}
.fh5co-form .form-control:focus, .fh5co-form .form-control:active {
  border-bottom: 1px solid rgba(0, 0, 0, 0.4);
}
/*
.btn-primary {
  height: 50px;
  padding-right: 20px;
  padding-left: 20px;
  border: none;
  background: #33cccc;
  color: #ffffff;
  -webkit-box-shadow: -2px 10px 20px -1px rgba(51, 204, 204, 0.4);
  -moz-box-shadow: -2px 10px 20px -1px rgba(51, 204, 204, 0.4);
  -o-box-shadow: -2px 10px 20px -1px rgba(51, 204, 204, 0.4);
  box-shadow: -2px 10px 20px -1px rgba(51, 204, 204, 0.4);
}
.btn-primary:hover, .btn-primary:focus, .btn-primary:active {
  color: #ffffff;
  background: #47d1d1 !important;
  outline: none;
}
*/
/* 
input, textarea {
  color: #000;
}

.placeholder {
  color: #aaa;
}

.js .animate-box {
  opacity: 0;
}

*/ 


//js
 
*/ 
