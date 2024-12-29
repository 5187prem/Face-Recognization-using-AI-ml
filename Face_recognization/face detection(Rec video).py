import cv2  # Computer vision library to read images and videos

alg = "haarcascade_frontalface_default.xml"  # Path to Haar Cascade XML file
haar_cascade = cv2.CascadeClassifier(alg)  # Loading Haar Cascade algorithm

video_path = r"samplevideo.mp4"  # Path to your video file
cam = cv2.VideoCapture(video_path)  # Load the video

while True:
    ret, img = cam.read()  # Read a frame from the video
    if not ret:  # If no frame is read (end of video), break the loop
        break

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale

    face = haar_cascade.detectMultiScale(grayImg, 1.3 , 8)  # Detect faces

    for (x, y, w, h) in face:  # Draw rectangles around detected faces
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Resize the frame to fit within the screen
    resized_img = cv2.resize(img, (800, 600))  # Resize to 800x600 pixels

    cv2.imshow("Face Detection", resized_img)  # Display the resized frame with detected faces

    key = cv2.waitKey(10)  # Wait for a key press
    if key == 27:  # Exit if the ESC key is pressed
        break

cam.release()  # Release the video file
cv2.destroyAllWindows()  # Close all OpenCV windows
