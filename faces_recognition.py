import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.xml')

font = cv2.FONT_HERSHEY_COMPLEX
txt_stroke = 1
rec_stroke = 1

labels = {}

# ラベル名の取得
with open('labels.pickle', 'rb') as f:
  label_dat = pickle.load(f)
  labels = {v:k for k, v in label_dat.items()}


cap = cv2.VideoCapture(0) # カメラから映像を取得

while(True):
  _, frame = cap.read() # 1フレーム毎に取得
  
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5) # 顔検出
  
  for (x, y, w, h) in faces:
    face_area = gray[y:y+h, x:x+w]

    idx, confidence = recognizer.predict(face_area) # 顔の識別
    # print(idx, confidence)
    
    name = 'unregistered'
    txt_color = (0, 0, 255)
    rec_color = (0, 0, 255)
    if confidence < 65: # 認証された場合 (0に近づくほど一致していることを表す)
      txt_color = (255, 255, 255)
      rec_color = (255, 0, 0)
      name = labels[idx]

    width = x + w
    height = y + h

    cv2.putText(frame, name, (x, y-10), font, 1, txt_color, txt_stroke, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (width, height), rec_color, rec_stroke)

  
  cv2.imshow('frame', frame) # 処理したフレームの表示
  
  # 終了条件
  if cv2.waitKey(24) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()
