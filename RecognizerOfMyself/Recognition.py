import cv2

rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('trainer/trainer.yml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

font = cv2.FONT_HERSHEY_SIMPLEX


id = 0

names = ['None', 'Mishanya', 'Katya', 'Vlada', 'Sasha', 'Alexey']


cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, conf = rec.predict(gray[y:y + h, x:x + w])

        if (conf < 100):
            id = names[id]
            conf = "  {0}%".format(round(100 - conf))
        else:
            id = "unknown person"
            confidence = "  {0}%".format(round(100 - conf))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(conf), (x + 5, y + h - 5), font, 1, (255, 255, 0), 2)

    cv2.imshow('Recognition', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break


cam.release()
cv2.destroyAllWindows()