import cv2  # Computer vision library to read and process images


alg = "haarcascade_frontalface_default.xml"  # Path to Haar Cascade XML file
haar_cascade = cv2.CascadeClassifier(alg)  # Loading Haar Cascade algorithm

image_path = "sampleimg.jpg"  # Replace with the path to your image file
img = cv2.imread(image_path)  # Load the image

# Resize the image to fit the screen while maintaining aspect ratio
screen_width = 800  # Set your screen width
screen_height = 600  # Set your screen height
height, width = img.shape[:2]
scale = min(screen_width / width, screen_height / height)
new_width = int(width * scale)
new_height = int(height * scale)
resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

grayImg = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)  # Convert the resized image to grayscale

faces = haar_cascade.detectMultiScale(grayImg, 1.3 ,2 )  # Detect faces

for (x, y, w, h) in faces:  # Draw rectangles around detected faces
    cv2.rectangle(resized_img, (x, y), (x + w, y + h), (0, 255, 0) , 2)

cv2.imshow("Face Detection", resized_img)  # Display the resized image with detected faces

cv2.waitKey(0)  # Wait indefinitely for a key press
cv2.destroyAllWindows()  # Close all OpenCV windows
