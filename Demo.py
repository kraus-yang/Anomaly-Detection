from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.regularizers import l2
from keras.optimizers import SGD ,Adagrad
from scipy.io import loadmat, savemat
from keras.models import model_from_json
import theano.tensor as T
import theano
import csv
import configparser
import collections
import time
import csv
from math import factorial
import os
from os import listdir
import skimage.transform
from skimage import color
from os.path import isfile, join
import numpy as np
import numpy.matlib
from datetime import datetime
from scipy.spatial.distance import cdist,pdist,squareform
import theano.sandbox
import time
#import c3D_model
#import Initialization_function
#from moviepy.editor import VideoFileClip
#from IPython.display import Image, display
import matplotlib.pyplot as plt
import cv2
import os, sys,shutil
import pickle
sys.path.append('/home/kraus/PycharmProjects/Real-World-Anomaly-Detection/C3D-v1.0/examples/')
import c3d_feature_extraction.extract_C3D_feature as ext


#from C3D-v1.0.examples.c3d_feature_extraction.exteact_C3D_feature  import_ extract_c3d_features_from_video
from PyQt5 import QtWidgets  # If PyQt4 is not working in your case, you can try PyQt5,
seed = 7
numpy.random.seed(seed)

def load_model(json_path):
    model = model_from_json(open(json_path).read())
    return model

def load_weights(model, weight_path):
    dict2 = loadmat(weight_path)
    dict = conv_dict(dict2)
    i = 0
    for layer in model.layers:
        weights = dict[str(i)]
        layer.set_weights(weights)
        i += 1
    return model

def conv_dict(dict2): # Helper function to save the model
    i = 0
    dict = {}
    for i in range(len(dict2)):
        if str(i) in dict2:
            if dict2[str(i)].shape == (0, 0):
                dict[str(i)] = dict2[str(i)]
            else:
                weights = dict2[str(i)][0]
                weights2 = []
                for weight in weights:
                    if weight.shape in [(1, x) for x in range(0, 5000)]:
                        weights2.append(weight[0])
                    else:
                        weights2.append(weight)
                dict[str(i)] = weights2
    return dict


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    #try:
    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    #except ValueError, msg:
    #    raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")

    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)

    half_window = (window_size - 1) // 2
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y,mode='valid')
# Load Video

def load_dataset_One_Video_Features(Test_Video_Path):

    VideoPath =Test_Video_Path
    csv_file = open(VideoPath)
    csv_reader_lines = csv.reader(csv_file)
    date_PyList = []
    for one_line in csv_reader_lines:
        date_PyList.append(one_line)
    AllFeatures = np.array(date_PyList)

    return AllFeatures


def make_prediton(video_path):
    Model_dir = '/home/kraus/PycharmProjects/Real-World-Anomaly-Detection/'
    weights_path = Model_dir + 'weights_L1L2.mat'
    model_path = Model_dir + 'model.json'
    global model
    model = load_model(model_path)
    load_weights(model, weights_path)




def Anomaly_detect(videos_dir):
    features_dir = '/home/kraus/PycharmProjects/Real-World-Anomaly-Detection/Videos_Features/'
    videoslst=os.listdir(videos_dir)
    for video_name in videoslst:
        video_path=videos_dir+video_name
        print(video_path)

        # Extract C3D features
        ext.c3d_extractor(video_path,features_dir)


        cap = cv2.VideoCapture(video_path)
        #Total_frames = cap.get(cv2.CV_CAP_PROP_FRAME_COUNT)
        print(cv2)
        Total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_segments = np.linspace(1, Total_frames, num=33)
        total_segments = total_segments.round()
        feature_name = video_name[0:-4]
        feature_name=feature_name+'.csv'
        FeaturePath = features_dir+ feature_name
        make_prediton(video_path)
        inputs = load_dataset_One_Video_Features(FeaturePath)
        #inputs = np.reshape(inputs, (32, 4096))
        predictions = model.predict_on_batch(inputs)
        danger=True if max(predictions)>=0.8 else False
        if danger:
            print ('[WARNING] There are dangerous incidents now !!!\nTime:{0}'.format(time.ctime()))
            #os.system('pause')
        Frames_Score = []
        count = -1;
        for iv in range(0, 32):
            F_Score = np.matlib.repmat(predictions[iv],1,(int(total_segments[iv+1])-int(total_segments[iv])))
            count = count + 1
            if count == 0:
              Frames_Score = F_Score
            if count > 0:
              Frames_Score = np.hstack((Frames_Score, F_Score))



        cap = cv2.VideoCapture((video_path))
        while not cap.isOpened():
            cap = cv2.VideoCapture((video_path))
            cv2.waitKey(1000)
            print ("Wait for the header")

        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        Total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        print ("Anomaly Prediction")
        x = np.linspace(1, Total_frames, Total_frames)
        scores = Frames_Score
        scores1=scores.reshape((scores.shape[1],))
        scores1 = savitzky_golay(scores1, 101, 3)
        plt.close()
        break_pt=min(scores1.shape[0], x.shape[0])
        plt.axis([0, Total_frames, 0, 1])
        i=0;
        while True:
            flag, frame = cap.read()
            if flag:
                i = i + 1
                #time.sleep(0.04)
                cv2.imshow('video', frame)
                jj=i%25
                if jj==1:
                    plt.plot(x[:i], scores1[:i], color='r', linewidth=3)
                    plt.draw()
                    plt.pause(0.000000000000000000000001)

                pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                print (str(pos_frame) + " frames")
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)

            if cv2.waitKey(10) == 27:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES)== break_pt:
                #cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break
        cap.release()
        cv2.destroyWindow('video')
        plt.close()
        #move the video files
        if  danger:
            danger_dir=videos_dir[:-1]+'_Dangerous/'+video_name
            shutil.move(video_path,danger_dir)
        else:
            normal_dir=videos_dir[:-1]+'_Normal/'+video_name
            shutil.move(video_path, normal_dir)
        #time.sleep(10)




def video_recording(videos_dir):
    ## opening videocapture
    cap = cv2.VideoCapture(0)
    print(cap)
    ## some videowriter props
    sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    fps = 20
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', 'e', 'g')
    # fourcc = cv2.VideoWriter_fourcc(*'mpeg')
    start_time = time.time()
    a = time.localtime(start_time)
    # print(a)
    b = ''
    for term in a:
        b = b + str(term) + '_'
    b = b[:-8]
    ## open and set props
    vout = cv2.VideoWriter()
    save_dir = videos_dir + b + '.mp4'
    #print(save_dir)
    vout.open(save_dir,fourcc,fps,sz,True)
    cnt = 0
    picture_recording=True
    print('Geting video from camera!')


    while 1:
        if int(time.time()) % 10 == 0:
            print('Time:{0}'.format(time.ctime()))
        _, frame = cap.read()
        cv2.putText(frame, time.ctime(), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv2.LINE_AA)
        vout.write(frame)
        if time.time()-start_time>=60:
            break
    vout.release()
    cap.release()




def main():
    videos_dir = '/home/kraus/PycharmProjects/Real-World-Anomaly-Detection/Videos/'
    start_t=time.time()
    print('Detection start!')
    while 1:
        # video_recording(videos_dir)
        Anomaly_detect(videos_dir)
        if time.time()-start_t>=300:
            print('Dectetion over!')
            break





if __name__== '__main__':
    main()