#camera.pyの置き場は　/home/pi/kadaiとする　（kadaiフォルダは作成してください）
#必要なモジュールのインポート（opencv , 時間扱うモジュール）

import cv2

import datetime#(発展で必要)
import os #(発展で必要)

#opencvでカメラ読み込み設定
#縦480,横640,15FPS
#画像を表示するwindowの名前はcapture

width=640
height=480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS,15)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 
cv2.namedWindow('capture') #キャプチャwindow設定

#メイン文の作成　
#while文などを利用して、カメラ画像をリアルタイムに映し出すコードを作成
#次にキーボードの"s"キーを押すと　"/home/pi/kadai/capture"というフォルダに画像を保存するコードを追加
#(発展)現在時刻を取得して日付フォルダを自動生成して　その中にファイルを保存するコードに書き換える

#連番用 i 初期化
i = 0
save_dir = './capture'
#save_dir = './capture'

while True :

    #画面モニタリング
    _,img=cap.read()
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
        cv2.imwrite(save_dir + folderdate + '{0:03d}'.format(i) +'.png',img)
        i+=1
    elif k == ord('q'):
        break

        
#終了処理　opencvのwindowを消す　（消さないと動作不安定に）
cv2.destroyAllWindows()