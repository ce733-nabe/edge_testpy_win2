#camera.pyの置き場は　/home/pi/kadaiとする　（kadaiフォルダは作成してください）
#必要なモジュールのインポート（opencv , 時間扱うモジュール）
import os
try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(os.getcwd())
except:
    pass

import cv2
import datetime#(発展で必要)

#機械学習部分を追加
import numpy as np
import pandas as pd
#import keras
#from keras.preprocessing.image import array_to_img, img_to_array, load_img
#from keras.models import load_model
#from keras.utils import np_utils
from sklearn.model_selection import train_test_split

import tensorflow as tf

#モデルロード
model = tf.keras.models.load_model('./model/t-model.h5')

#opencvでカメラ読み込み設定
#縦480,横640,15FPS
#画像を表示するwindowの名前はcapture

width=640
height=480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,15)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 
cv2.namedWindow('capture',cv2.WINDOW_NORMAL) #キャプチャwindow設定

#メイン文の作成　
#while文などを利用して、カメラ画像をリアルタイムに映し出すコードを作成
#次にキーボードの"s"キーを押すと　"/home/pi/kadai/capture"というフォルダに画像を保存するコードを追加
#(発展)現在時刻を取得して日付フォルダを自動生成して　その中にファイルを保存するコードに書き換える

#連番用 i 初期化
i = 0
save_dir = './capture'

x_zahyo = 50
y_zahyo = 50
while True :

    #画面モニタリング
    _,img=cap.read()
    #cv2.rectangle(img, (x_zahyo, y_zahyo), (width-x_zahyo, height-y_zahyo), (0, 0, 255))
    cv2.imshow('capture',img)
    k = cv2.waitKey(1)&0xFF


    #(発展)現在日時から日付フォルダを作成
    captime=datetime.datetime.now()
    folderdate = '/{0:%Y%m%d}/'.format(captime)
    os.makedirs(save_dir+folderdate, exist_ok=True)

    if k == 27:         # 27はesc
        break
        cv2.destroyAllWindows()
    elif k == ord('s'): # 's'
        #opencv(BGR配列)をpillow形式(RGB)に変換して
        img = cv2.resize(img, dsize=(64, 64))
        img = img[:, :, ::-1]
        
        X = img[np.newaxis,:,:,:]
        #print(X.shape)

        # 画素値を0から1の範囲に変換(正規化)
        X = X.astype('float32')/255

        prediction = model.predict(X)

        print(prediction)
        img2 = np.ascontiguousarray(img, dtype=np.uint8)
        if prediction[0][0]>0.8:
            print('ボルト銀')
            cv2.putText(img2, 'bolt_gin', (20,20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 1, cv2.LINE_AA)

        elif prediction[0][1]>0.8:
            print('ボルト黒')
            cv2.putText(img2, 'bolt_kuro', (20,20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 1, cv2.LINE_AA)
        else:
            print('判別不明')
            cv2.putText(img2, 'non', (20,20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255,0), 1, cv2.LINE_AA)
        cv2.imshow('capture2',img2)
        
        cv2.imwrite(save_dir + folderdate + '{0:03d}'.format(i) +'.png',img)
        i+=1
        
    elif k == ord('q'):
        break
        
#終了処理　opencvのwindowを消す　（消さないと動作不安定に）
cv2.destroyAllWindows()
