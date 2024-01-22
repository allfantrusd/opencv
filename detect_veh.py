import cv2

# Load pre-trained Haar Cascade classifier for vehicles
vehicle_cascade = cv2.CascadeClassifier('car.xml')

# Open the video file
vid = cv2.VideoCapture("Videos/car2.mp4")
list_count = []
while True:
    # Read a frame from the video
    is_T, frame = vid.read()

    # Break the loop if the video is finished
    if not is_T:
        break

    # Convert the frame to grayscale (Haar Cascade works on grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform vehicle detection
    vehicles = vehicle_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=9)

    count = 0

    for (x, y, w, h) in vehicles:
        count +=1
        if count not in list_count:
            list_count.append(count)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        text = f"Vehicle {count}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

    # Display the frame with bounding boxes around detected vehicles
    cv2.imshow("Vehicle Detection", frame)

    # Break the loop if 'd' key is pressed
    if cv2.waitKey(10) & 0xFF == ord('d'):
        break
print(list_count)
# Release the video capture object and close all windows
vid.release()
cv2.destroyAllWindows()
