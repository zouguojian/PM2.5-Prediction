# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 23:35:58 2018

@author: Administrator
"""
import tensorflow as tf
import random
import ResCNN
import CV_LSTM
import TrainData
import TestData
import numpy as np
import matplotlib.pyplot as plt
import datetime
tf.reset_default_graph()
batch_size=32
#初始化batch_size大小
all_around_station=14
time_lag=3
#预测未来的时刻
TIME_SIZE=20
#过去的时间序列
IMAGE_HIGHTH=14
IMAGE_WIDTH=7
NUM_CHANNELS=1
OUT_NUM=1
LEARNING_RATE=0.0005

epochs=101
#a=np.array([[[[1,2,3,4],[5,6,7,8]]]])
def train():
    SH_train=TrainData.Dataset()
    SH_test=TestData.Dataset()
    X=tf.placeholder(tf.float32,[TIME_SIZE*batch_size,IMAGE_HIGHTH,IMAGE_WIDTH,
                                 NUM_CHANNELS],name="x_input")
    Y= tf.placeholder(tf.float32, [None,OUT_NUM])
    cnn=ResCNN.ResCNN(X,batch_size)
    result=cnn.CNN_layer()
#    result=X
    print(result.shape)
    '''
    经过卷积神经网络的输出
    '''
#    shape=np.array([[None]*result.shape[1] for i in range(result.shape[2])])
    shape=np.array([[None]*result.shape[2] for i in range(result.shape[1])])
#状态的大小，即高和宽
    filter_size=np.array([[None]*2 for i in range(2)])
#卷积核的大小，长和宽的大小
    num_features=result.shape[3]
#初始的通道数，即输入和状态的通道数
    c_lstm=CV_LSTM.BasicConvLSTMCell(shape.shape,filter_size.shape,num_features,TIME_SIZE)
    state=c_lstm.zero_state(batch_size)
#第一次状态值的初始化
    inputs=tf.reshape(result,[-1,TIME_SIZE,shape.shape[0],shape.shape[1],num_features])
    y=c_lstm.Full_connect(inputs,state)
    cross_entropy=tf.sqrt(tf.reduce_mean(tf.square(y-Y)))
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    
    SH_train_low=0*all_around_station
    SH_train_high=SH_train_low+batch_size*all_around_station*TIME_SIZE
    '''
    经过卷积LSTM的输出
    '''
    saver=tf.train.Saver(max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        start_time=datetime.datetime.now()
        average=5000
        for e in range(epochs):

            while(SH_train_high<SH_train.shape[0]):
                train_x=SH_train[SH_train_low:SH_train_high,2:9]
#                train_x=(train_x-train_x.min())/(train_x.max()-train_x.min())
                reshaped_x = np.reshape(train_x, (
                        TIME_SIZE*batch_size,
                        IMAGE_HIGHTH,
                        IMAGE_WIDTH,
                        NUM_CHANNELS))
                label=list()
                for line in range(batch_size):
                    if str(SH_train[SH_train_low+(TIME_SIZE*(line+1))*all_around_station+time_lag*all_around_station][1])=='ShangHai':
                        label.append(SH_train[SH_train_low+(TIME_SIZE*(line+1))*all_around_station+time_lag*all_around_station,3:4])
                label=np.array(label)
                s,_=sess.run((cross_entropy,train_op), feed_dict={X:reshaped_x,Y:label})
#                测试模型预测效果
                SH_train_low=SH_train_high
                SH_train_high=SH_train_low+batch_size*all_around_station*TIME_SIZE
            SH_train_low=random.randint(0, 0)*all_around_station
#            SH_train_low=random.randint(0, 10)*all_around_station  
            SH_train_high=SH_train_low+batch_size*all_around_station*TIME_SIZE
            #test procession
            SH_test_low=0*all_around_station
            SH_test_high=SH_test_low+batch_size*all_around_station*TIME_SIZE
            if e%1==0:
                Label=list()
                Predict=list()
                while(SH_test_high<SH_test.shape[0]):
                    test_x=SH_test[SH_test_low:SH_test_high,2:9]
#                    test_x=(test_x-test_x.min())/(test_x.max()-test_x.min())
                    reshaped_x = np.reshape(test_x, (
                                    TIME_SIZE*batch_size,
                                    IMAGE_HIGHTH,
                                    IMAGE_WIDTH,
                                    NUM_CHANNELS))
                    label=list()
                    for line in range(batch_size):
                        if str(SH_test[SH_test_low+(TIME_SIZE*(line+1))*all_around_station+time_lag*all_around_station][1])=='ShangHai':
                            label.append(SH_test[SH_test_low+(TIME_SIZE*(line+1))*all_around_station+time_lag*all_around_station,3:4])
                    label=np.array(label)
                    s=sess.run((y), feed_dict={X:reshaped_x})
                    s=np.reshape(s,[1,batch_size])[0]
                    for i in range(batch_size):
                        Label.append(float(label[i]))
                        Predict.append(s[i])
                    SH_test_low=SH_test_high
                    SH_test_high=SH_test_low+batch_size*all_around_station*TIME_SIZE
                Label=np.reshape(np.array(Label),[1,-1])[0]
                Predict=np.array(Predict)
                error=Label-Predict
                average_Error=np.mean(np.fabs(error))
                print("After %d steps, MAE error is : %f"%(e,average_Error))
#                            print(test_y_)
                RMSE_Error=np.sqrt(np.mean(np.square(np.array(Label)-np.array(Predict))))
                if average>RMSE_Error:
                    print('average has been changed!')
                    average=RMSE_Error
                    saver.save(sess,'C:/Users/butany/Desktop/experiment/NewRCL/model/model.ckpt',global_step=time_lag+1)
                print("After %d steps, RMSE error is : %f"%(e,RMSE_Error))
                cor=np.mean(np.multiply((Label-np.mean(Label)),
                                                   (Predict-np.mean(Predict))))/(np.std(Predict)*np.std(Label))
                print ('The correlation coefficient is: %f'%(cor))
                if e==10 or e==20 or e==30 or e==40 or e==50 or e==60 or e==70 or e==80 or e==90 or e==100:
                        plt.figure() 
#                Label是真实值,蓝色的为真实值
                        plt.plot(Label,'b*:',label=u'observed value')  
#                Predict是预测值，红色的为预测值
                        plt.plot(Predict,'r*:',label=u'predicted value')
#                       让图例生效
                        plt.legend()
                        plt.xlabel("Time(hours)",fontsize=17)  
                        plt.ylabel("PM2.5(ug/m3)",fontsize=17)  
                        plt.title("The prediction of PM2.5  (epochs ="+str(e)+")",fontsize=17)
            
#            运行时间的差
        end_time=datetime.datetime.now()
        total_time=end_time-start_time
        print("Total runing times is : %f"%total_time.total_seconds())           
 
def main(argv=None):
    train()

if __name__ == '__main__':
    main()
