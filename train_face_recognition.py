import cv2
import os
from PIL import Image
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(BASE_DIR, 'train_images')

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(img_dir):
  for file in files:
    if file.endswith('png') or file.endswith('jpg'):
      path = os.path.join(root, file)
      label = os.path.basename(root).replace(" ", "-").lower() # 一番下のディレクトリ名がラベルの名前となる
      
      # ラベルのリストを作成
      if not label in label_ids:
        label_ids[label] = current_id
        current_id += 1
      
      idx = label_ids[label]
      pil_img = Image.open(path).convert('L') # グレースケール化
      img_size = (550, 550)
      fin_img = pil_img.resize(img_size, Image.ANTIALIAS) # 各写真のサイズを統一
      img_array = np.array(fin_img, 'uint8') # 8ビットの行列として画像(fin_img)を表す
      faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5) # 顔検出
      
      # 訓練画像中において，顔が検出されなかったとき
      if not len(faces):
        print('以下の画像では，顔の検出が行われませんでした．')
        print('>>>> ', path)
        pass

      for (x, y, w, h) in faces:
        face_area = img_array[y:y+h, x:x+w]
        x_train.append(face_area)
        y_labels.append(idx)

# ラベル名の保存
with open('labels.pickle', 'wb') as f:
  pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainner.xml')

