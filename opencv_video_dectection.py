import cv2

# Load different cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smile.xml")


# To capture video from webcam. 
cap = cv2.VideoCapture(0)

# To use a video file as input 
# cap = cv2.VideoCapture('filename.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        # Detect eyes
        eyes=eye_cascade.detectMultiScale(gray, 1.1, 3)
    
        # Draw rectangle around the eyes
        for (a, b, c, d) in eyes:
            cv2.rectangle(img, (a, b), (a + c, b + d), (255, 0, 0), 2)
    
        # Detect smiles
        smiles=smile_cascade.detectMultiScale(gray, 1.4, 40)
    
        # Draw rectangle around the smiles
        for (m, n, o, p) in smiles:
            cv2.rectangle(img, (m, n), (m + o, n + p), (0, 0, 255), 2)

        

    # Display
    cv2.imshow('img', img)

    
    # Stop if escape key or 'q' key is pressed
    if cv2.waitKey(30) == ord('q') or cv2.waitKey(30)==27:
        break
        
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()
