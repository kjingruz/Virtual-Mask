import cv2
import dlib
import numpy as np

# Load the pre-trained models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
predictor_model = "shape_predictor_68_face_landmarks.dat"  # You'll need to download this model
face_predictor = dlib.shape_predictor(predictor_model)
face_detector = dlib.get_frontal_face_detector()

# Function to apply a Batman mask
def apply_batman_mask(frame, face_rect, landmarks, mask_path='batman.png'):
    # Load the mask image
    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask_height, mask_width, _ = mask_img.shape

    # Calculate the size and position of the mask
    face_width = face_rect.right() - face_rect.left()
    
    # The factor 1.5 below is adjustable to ensure the mask covers the whole face width
    scale = (face_width / mask_width) * 1.5
    mask_img = cv2.resize(mask_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    new_mask_height, new_mask_width, _ = mask_img.shape

    # Calculate the position to place the mask
    x_center = (face_rect.left() + face_rect.right()) // 2
    y_center = (face_rect.top() + face_rect.bottom()) // 2

    x1 = x_center - new_mask_width // 2
    x2 = x1 + new_mask_width
    y1 = y_center - new_mask_height // 2  # This centers the mask vertically
    y2 = y1 + new_mask_height

    # Extract the alpha channel from the mask image
    mask_alpha = mask_img[:, :, 3] / 255.0
    img_alpha = 1.0 - mask_alpha

    # Ensure the mask fits within the frame boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    # Check if the mask goes out of frame boundaries
    if x1 >= x2 or y1 >= y2:
        return

    mask_alpha = mask_alpha[(y1 - y_center + new_mask_height // 2):(y2 - y_center + new_mask_height // 2), 
                            (x1 - x_center + new_mask_width // 2):(x2 - x_center + new_mask_width // 2)]
    img_alpha = 1.0 - mask_alpha

    # Replace the pixels with the mask
    for c in range(0, 3):
        frame[y1:y2, x1:x2, c] = (mask_alpha * mask_img[(y1 - y_center + new_mask_height // 2):(y2 - y_center + new_mask_height // 2), 
                                                        (x1 - x_center + new_mask_width // 2):(x2 - x_center + new_mask_width // 2), c]) + \
                                 (img_alpha * frame[y1:y2, x1:x2, c])

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        face_rect = dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h)
        landmarks = face_predictor(gray, face_rect)

        # Apply Batman mask
        apply_batman_mask(frame, face_rect, landmarks)

    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
