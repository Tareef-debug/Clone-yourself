import os
import time
import cv2
import numpy as np
import onnxruntime as ort

# Load the model from the same folder as this Python file
folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(folder, "modnet.onnx")

model = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

input_name = model.get_inputs()[0].name
shape = model.get_inputs()[0].shape

height = shape[2] if isinstance(shape[2], int) else 512
width = shape[3] if isinstance(shape[3], int) else 512

os.makedirs("photos", exist_ok=True)

camera = cv2.VideoCapture(0)
background = None
timer = None

if not camera.isOpened():
    raise RuntimeError("Could not open camera")

try:
    while True:
        success, frame = camera.read()

        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)

        # Prepare the image for the AI
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        image = image.astype(np.float32) / 127.5 - 1
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)

        # Get the person's matte
        result = model.run(None, {input_name: image})
        matte = result[0].squeeze()
        matte = cv2.resize(matte, (640, 480))
        matte = np.clip(matte, 0, 1)
        matte = matte[:, :, None]

        photo = frame.copy()

        # Draw the live person in front of the frozen copies
        if background is not None:
            photo = frame * matte + background * (1 - matte)
            photo = photo.astype(np.uint8)

        # Freeze the pose after three seconds
        if timer is not None:
            remaining = 3 - int(time.monotonic() - timer)

            if remaining <= 0:
                background = photo.copy()
                timer = None
                print("Copy frozen! Move to another position.")

        preview = photo.copy()

        # Show the countdown without adding it to the saved photo
        if timer is not None:
            cv2.putText(
                preview, str(remaining), (280, 240),
                cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                4, (255, 255, 0), 6
            )

        cv2.imshow("Live camera", preview)
        key = cv2.waitKey(1) & 0xFF

        # Capture the empty room
        if key == ord("b"):
            background = frame.copy()
            timer = None
            print("Background captured")

        # Start the countdown
        elif key == ord("c"):
            if background is not None and timer is None:
                timer = time.monotonic()
            elif background is None:
                print("Capture the background first")

        # Save the combined photo
        elif key == ord("s"):
            filename = os.path.join(
                "photos", f"capture_{time.time_ns()}.png"
            )

            if cv2.imwrite(filename, photo):
                print("Saved:", filename)
            else:
                print("Could not save the file")

        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()