import cv2 #computer vision library to read images

alg = ""

haar_cascade = cv2.CascadeClassifier(alg)  #loading algorithm

cam = cv2.VideoCapture(1)  #cam id intialization

while True:
    _,img = cam.read() #reading the frame from camera

    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converting clear img to gray

    face = haar_cascade.detectMultiScale(grayImg,1.3,4)  #getting coordinates

    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.imshow("FaceDectection",img)

    key = cv2.WaitKey(10)
    print(key)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()

