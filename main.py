import cv2
# Direct imports from MediaPipe
import mediapipe.python.solutions.hands as mp_hands
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.drawing_styles as mp_drawing_styles

def run_hand_tracking():
    cam = cv2.VideoCapture(0)

    # Initialize MediaPipe Hands
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while cam.isOpened():
            success, frame = cam.read()
            if not success:
                print("Empty frame")
                continue

            #Flip the frame for natural mirror effect
            frame = cv2.flip(frame, 1)

            #Converting Colour Format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            #Draw landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                    )

            #Show the frame
            cv2.imshow("Hand Tracking", frame)

            #Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Clean up
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hand_tracking()

