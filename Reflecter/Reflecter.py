#!usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import tkinter as Tk
import numpy as np
import cv2
import pyautogui as pg
from PIL import Image, ImageTk
from keras.preprocessing import image
from keras.models import model_from_json
from keras.utils import to_categorical
import time
from concurrent.futures import ThreadPoolExecutor
running = False

# predict及びmode
# 0:RIGHT
# 1:LEFT
# 2:SQUAT
# 3:STOP
# 4:JUMP
# 5:FIRE

def right():
    pg.keyDown('d')
    print('RIGHT')

def left():
    pg.keyDown('a')
    print('LEFT')

def squat():
    pg.keyDown('s')
    print('SQUAT')

def stop():
    pg.keyUp('a')
    pg.keyUp('d')
    pg.keyUp('s')
    pg.keyUp('shift')
    print('STOP')

def jump():
    for i in range(15):
        pg.keyDown('w')
    pg.keyUp('w')
    print('JUMP')

def fire():
    pg.keyDown('shift')
    print('FIRE')

def rightjump():
    for i in range(15):#ジャンプと右　同時
        pg.keyDown('d')
        pg.keyDown('w')
    pg.keyUp('w')
    print('RIGHT_JUMP')

def leftjump():
    for i in range(15):#ジャンプと右　同時
        pg.keyDown('a')
        pg.keyDown('w')
    pg.keyUp('w')
    print('LEFT_JUMP')

#reflect
def reflect(event):
    running = True
    print('---START---')

    #10回に一回動くカウントを宣言
    count = 0
    #move,modeを初期化
    move = ''
    mode = -1
    #並列化処理
    pool = ThreadPoolExecutor(4)
    #VideoCaptureオブジェクトを取得
    cap = cv2.VideoCapture(0)
    # モデルの読み込み
    model = model_from_json(open('9767.json', 'r').read())
    # 重みの読み込み
    model.load_weights('9767.h5')
    #画像認識開始
    while True:
        start = time.time()
         #画像を読み込み
        ret,frame = cap.read()
        edframe = frame
        cv2.putText(edframe, move, (0,50), cv2.FONT_HERSHEY_COMPLEX_SMALL | cv2.FONT_ITALIC,3,(100,200,255),2,cv2.LINE_AA)
        #フレーム画像を表示
        cv2.imshow("frame",edframe)
        #retがなかったら終了
        if not ret:
            print('error')
            break
        #もしqが押されたら、終わり
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #カウントをインクリメント
        count += 1
        #カウントが10貯まったら
        if count == 10:
            #画像をコピー
            img = frame.copy()
            #画像をリサイズ
            img = cv2.resize(img, (224, 224))
            #kerasが読み込みやすいようにする。[1,224,224,3]
            img = image.img_to_array(img)
            img = img/255
            img = np.expand_dims(img, axis=0)
            #モデルをつかって予想　predictは予想の値
            predicts = model.predict(img)
            predict = np.argmax(predicts)
            #実際の動きに変換し、コントローラを動かす
            #ジャンプ
            if predict == 4:
                move = 'JUMP'
                if mode == 0:
                    pool.submit(rightjump)
                elif mode == 1:
                    pool.submit(leftjump)
                else:
                    pool.submit(jump)
            else:
                if predict == 0:
                    move = 'RIGHT'
                    pool.submit(right)
                    mode = 0
                elif predict == 1:
                    move = 'LEFT'
                    pool.submit(left)
                    mode = 1
                elif predict == 2:
                    move = 'SQUAT'
                    pool.submit(squat)
                    mode = -1
                elif predict == 3:
                    move = 'STOP'
                    pool.submit(stop)
                    mode = -1
                elif predict == 5:
                    move = 'FIRE'
                    pool.submit(fire)
            end = time.time()
            print(end-start)
            #カウントを初期化
            count = 0

    #カメラ、ウィンドウを削除
    cap.release()
    cv2.destroyAllWindows()
    print('---END---')

if __name__ == '__main__':
    root = Tk.Tk()
    root.title(u'Reflecter')

    #画像をつくる
    off_image = cv2.imread('off.png')
    b,g,r = cv2.split(off_image)
    off_image = cv2.merge((r,g,b))
    off_im = Image.fromarray(off_image)
    off_imtk = ImageTk.PhotoImage(image=off_im)
    on_image = cv2.imread('on.png')
    b,g,r = cv2.split(on_image)
    on_image = cv2.merge((r,g,b))
    on_im = Image.fromarray(on_image)
    on_imtk = ImageTk.PhotoImage(image=on_im)

    # Put it in the display window
    button = Tk.Button(root, image=on_imtk)
    button.bind("<Button-1>",reflect)
    button.pack()

    root.mainloop()
