from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.template import loader
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from .models import Member
from .forms import SignUpForm
from django.utils.html import format_html
from PIL import Image, ImageDraw, ImageFont


from django.http import StreamingHttpResponse
import cv2
import mediapipe as mp
import numpy as np


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


# State Variables
calculator_display = ""
operation_complete = False
last_detected_gesture = None  # To manage debounce logic


# Calculation Logic
def perform_calculation(expression):
    """Safely evaluate mathematical expressions."""
    try:
        return str(eval(expression))  # Evaluate mathematical expressions
    except:
        return "Error"


# Detect interactions based on index/thumb proximity
def detect_gesture_proximity(hand_landmarks, buttons):
    """Detect hand gestures by index + thumb proximity for interaction."""
    global last_detected_gesture  # Use global to track last interaction
    
    if len(hand_landmarks) == 21:
        index_tip = hand_landmarks[8]
        thumb_tip = hand_landmarks[4]

        # Calculate their positions on the screen
        index_x, index_y = index_tip.x, index_tip.y
        thumb_x, thumb_y = thumb_tip.x, thumb_tip.y

        # Proximity detection logic
        if abs(index_x - thumb_x) < 0.05 and abs(index_y - thumb_y) < 0.05:  # Detect touch condition
            hand_screen_x = int(index_x * 640)  # Scale screen coordinates
            hand_screen_y = int(index_y * 480)
            for (button_x, button_y, label) in buttons:
                if abs(hand_screen_x - button_x) < 30 and abs(hand_screen_y - button_y) < 30:
                    # Avoid repeated gesture triggers using debounce logic
                    if label != last_detected_gesture:
                        last_detected_gesture = label  # Set the latest detected gesture
                        return label  # Detected new button click
    return None


# Draw interactive calculator buttons on video frame
def draw_calculator_ui(frame):
    """Draw calculator buttons inside a fixed and visible UI area."""
    # Right-hand side calculator UI positions
    buttons = [
        (400, 80, "1"), (470, 80, "2"), (540, 80, "3"),
        (400, 150, "4"), (470, 150, "5"), (540, 150, "6"),
        (400, 220, "7"), (470, 220, "8"), (540, 220, "9"),
        (400, 290, "0"), (470, 290, "+"), (540, 290, "-"),
        (400, 360, "*"), (470, 360, "/"),
        (540, 360, "="), (330, 360, "AC")
    ]

    # Render the buttons on the frame
    for (x, y, label) in buttons:
        cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), (200, 200, 255), -1)  # Button background
        cv2.putText(frame, label, (x - 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    return buttons


# Stream video frames with gesture handling
def generate_frames():
    """Main video feed loop with interaction support."""
    global calculator_display, operation_complete, last_detected_gesture
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            # Flip for a mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Overlay calculator buttons
            buttons = draw_calculator_ui(frame)

            # Handle hand gestures
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    detected_gesture = detect_gesture_proximity(hand_landmarks.landmark, buttons)

                    # Handle interaction logic
                    if detected_gesture:
                        if detected_gesture == "AC":
                            calculator_display = ""
                            operation_complete = False
                        elif detected_gesture == "=":
                            calculator_display = perform_calculation(calculator_display)
                            operation_complete = True
                        else:
                            if operation_complete:
                                calculator_display = detected_gesture
                                operation_complete = False
                            else:
                                calculator_display += detected_gesture

            # Display the current state of the calculator on top of the feed
            cv2.putText(frame, f"Expression: {calculator_display}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (180, 205, 155), 4)

            # Reset last gesture when hand is out of interaction range
            if not results.multi_hand_landmarks:
                last_detected_gesture = None

            # Stream the frame
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# Django view for serving the live camera feed
def calculator_view(request):
    """Serve video feed with the interactive calculator UI overlay."""
    return StreamingHttpResponse(
        generate_frames(),
        content_type="multipart/x-mixed-replace; boundary=frame"
    )
    
    
    
import threading
from django.shortcuts import redirect

# A flag to track whether the virtual volume control is running
volume_control_running = False

def trigger_volume_control(request):
    global volume_control_running
    if not volume_control_running:
        volume_control_running = True
        threading.Thread(target=vm).start()
    return redirect('homee')  # Redirect to home after starting volume control

def vm():
    import cv2
    import mediapipe as mp
    import numpy as np
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from comtypes import CLSCTX_ALL
    import math

    global volume_control_running

    # Initialize MediaPipe Hand model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Access system audio for volume control
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    vol_range = volume.GetVolumeRange()  # Volume range (-65.25, 0.0)

    min_vol = vol_range[0]  # Minimum volume
    max_vol = vol_range[1]  # Maximum volume

    # Stabilization variables
    prev_volume = None  # Keeps track of the last stable volume
    smoothing_frames = 5  # Number of frames for stability
    last_set_volume = 0  # Holds the last adjusted volume
    lock_threshold = 10  # Distance threshold to unlock volume control

    # Webcam capture
    cap = cv2.VideoCapture(0)
    stable_count = 0  # Counter for stability check

    while volume_control_running:
        success, img = cap.read()
        if not success:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        results = hands.process(img_rgb)  # Process the frame for hand detection

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmark positions for thumb (4) and index finger (8)
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                h, w, _ = img.shape  # Image dimensions
                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                # Calculate the distance between thumb and index finger
                distance = math.hypot(x2 - x1, y2 - y1)

                # Map the distance to volume range
                vol = np.interp(distance, [30, 200], [min_vol, max_vol])
                vol_percent = np.interp(vol, [min_vol, max_vol], [0, 100])

                # Check for stability - only update after a few stable frames
                if abs(vol_percent - last_set_volume) > lock_threshold:
                    stable_count += 1
                    if stable_count >= smoothing_frames:
                        last_set_volume = vol_percent
                        volume.SetMasterVolumeLevel(vol, None)
                        prev_volume = vol
                        stable_count = 0
                else:
                    stable_count = 0

                # Display volume level
                cv2.putText(img, f'Volume: {int(last_set_volume)}%', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the webcam feed
        cv2.imshow("Virtual Volume Control (Stable)", img)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Set the flag to False when the webcam loop is terminated
    volume_control_running = False

def stop_volume_control(request):
    global volume_control_running
    volume_control_running = False  # Stop the webcam thread
    return redirect('homee')  # Redirect to home after stopping volume control

    
    
    
    
import cv2
import mediapipe as mp
from pynput.mouse import Controller, Button
import numpy as np
import threading

def sm():
    # Initialize Mediapipe Hand Detection
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    

    # Initialize Mouse Controller
    mouse = Controller()

    # Screen dimensions (adjust based on your screen resolution)
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

    # Smoothing parameters for cursor stabilization
    prev_x, prev_y = 0, 0
    smooth_factor = 4

    # Scroll parameters
    scroll_threshold = 0.2  # Adjust as needed
    scroll_speed = 3  # Adjust scroll speed

    # Webcam capture
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame for a mirrored view
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Mediapipe
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Extract landmarks for the index finger tip and thumb tip
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Calculate screen coordinates for cursor control
                x = int(index_finger_tip.x * SCREEN_WIDTH)
                y = int(index_finger_tip.y * SCREEN_HEIGHT)

                # Smooth cursor movement
                curr_x = prev_x + (x - prev_x) // smooth_factor
                curr_y = prev_y + (y - prev_y) // smooth_factor
                prev_x, prev_y = curr_x, curr_y

                # Move the mouse to the calculated position
                mouse.position = (curr_x, curr_y)

                # Calculate the distance between the index finger and thumb
                thumb_x = int(thumb_tip.x * SCREEN_WIDTH)
                thumb_y = int(thumb_tip.y * SCREEN_HEIGHT)
                pinch_distance = np.sqrt((curr_x - thumb_x) ** 2 + (curr_y - thumb_y) ** 2)

                # If the distance is small, perform a click
                if pinch_distance < 70:  # Adjust the threshold as needed
                    mouse.click(Button.left, 1)

                # Scroll detection based on middle finger movement
                index_finger_y = index_finger_tip.y
                middle_finger_y = middle_finger_tip.y

                # Scroll up if middle finger is above index finger
                if (middle_finger_y - index_finger_y) > scroll_threshold:
                    mouse.scroll(0, scroll_speed)  # Scroll up
                # Scroll down if middle finger is below index finger
                elif (index_finger_y - middle_finger_y) > scroll_threshold:
                    mouse.scroll(0, -scroll_speed)  # Scroll down

                # Draw the hand landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the frame
        cv2.imshow('Virtual Mouse with Scroll', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def trigger_virtual_mouse(request):
    # Run the virtual mouse code in a separate thread
    threading.Thread(target=sm).start()
    return redirect('homee')




import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math
import time

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Access system audio for volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
vol_range = volume.GetVolumeRange()  # Volume range (-65.25, 0.0)

min_vol = vol_range[0]  # Minimum volume
max_vol = vol_range[1]  # Maximum volume

# Stabilization variables
prev_volume = None  # Keeps track of the last stable volume
smoothing_frames = 5  # Number of frames for stability
last_set_volume = 0  # Holds the last adjusted volume
lock_threshold = 10  # Distance threshold to unlock volume control

# Webcam capture
cap = cv2.VideoCapture(0)
stable_count = 0  # Counter for stability check

def virtual_volume_control(request):
    while True:
        success, img = cap.read()
        if not success:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        results = hands.process(img_rgb)  # Process the frame for hand detection

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmark positions for thumb (4) and index finger (8)
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                h, w, _ = img.shape  # Image dimensions
                x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

                # Draw circles and line between thumb and index finger
                cv2.circle(img, (x1, y1), 8, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 8, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Calculate the distance between thumb and index finger
                distance = math.hypot(x2 - x1, y2 - y1)

                # Map the distance to volume range
                vol = np.interp(distance, [30, 200], [min_vol, max_vol])
                vol_percent = np.interp(vol, [min_vol, max_vol], [0, 100])

                # Check for stability - only update after a few stable frames
                if abs(vol_percent - last_set_volume) > lock_threshold:
                    stable_count += 1
                    if stable_count >= smoothing_frames:
                        last_set_volume = vol_percent
                        volume.SetMasterVolumeLevel(vol, None)
                        prev_volume = vol
                        stable_count = 0
                else:
                    stable_count = 0

                # Display volume level
                cv2.putText(img, f'Volume: {int(last_set_volume)}%', (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the webcam feed
        cv2.imshow("Virtual Volume Control (Stable)", img)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render(request, 'homee')



from django.shortcuts import render, redirect
from django.http import HttpResponse
import cv2
import mediapipe as mp
import numpy as np

def virtual_drawing(request):
    # Initialize Mediapipe Hand Detection
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    # Canvas setup
    canvas_size = (640, 480)
    canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
    drawing_color = (255, 255, 255)  # Default drawing color: White
    brush_size = 10  # Brush size
    colors = {
        'White': (255, 255, 255),
        'Red': (0, 0, 255),
        'Green': (0, 255, 0),
        'Blue': (255, 0, 0),
        'Yellow': (0, 255, 255),
    }
    current_color = 'White'  # Default color

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    def draw_color_buttons(frame):
        """Draw color selection buttons on the frame."""
        button_radius = 25
        spacing = 70
        y_pos = 50
        for i, (color_name, color) in enumerate(colors.items()):
            x_pos = 50 + i * spacing
            cv2.circle(frame, (x_pos, y_pos), button_radius, color, -1)
            if color_name == current_color:
                cv2.circle(frame, (x_pos, y_pos), button_radius, (0, 0, 0), 2)  # Highlight selected color

    def draw_clear_button(frame):
        """Draw clear canvas button on the frame."""
        button_width = 100
        button_height = 40
        x_start = canvas_size[0] - button_width - 20
        y_start = canvas_size[1] - button_height - 20
        cv2.rectangle(frame, (x_start, y_start), (x_start + button_width, y_start + button_height), (0, 0, 255), -1)
        cv2.putText(frame, "Clear", (x_start + 10, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_home_button(frame):
        """Draw 'Back to Home' button on the frame."""
        button_width = 120
        button_height = 40
        x_start = 20
        y_start = canvas_size[1] - button_height - 20
        cv2.rectangle(frame, (x_start, y_start), (x_start + button_width, y_start + button_height), (0, 128, 0), -1)
        cv2.putText(frame, "Home", (x_start + 30, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    prev_point = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip the frame for mirrored view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        draw_color_buttons(frame)
        draw_clear_button(frame)
        draw_home_button(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                # Get the tip of the index finger
                index_finger_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_finger_tip.x * canvas_size[0]), int(index_finger_tip.y * canvas_size[1])

                # Drawing mode
                if prev_point:
                    cv2.line(canvas, prev_point, (x, y), drawing_color, brush_size)
                prev_point = (x, y)

                # Color selection
                for i, (color_name, color) in enumerate(colors.items()):
                    button_x = 50 + i * 70
                    button_y = 50
                    if np.linalg.norm(np.array([x, y]) - np.array([button_x, button_y])) < 25:
                        current_color = color_name
                        drawing_color = color

                # Clear canvas if clear button is pressed
                clear_button_start = (canvas_size[0] - 120, canvas_size[1] - 60)
                clear_button_end = (canvas_size[0] - 20, canvas_size[1] - 20)
                if clear_button_start[0] <= x <= clear_button_end[0] and clear_button_start[1] <= y <= clear_button_end[1]:
                    canvas.fill(0)  # Clear the canvas
                    prev_point = None

                # Back to Home if home button is pressed
                home_button_start = (20, canvas_size[1] - 60)
                home_button_end = (140, canvas_size[1] - 20)
                if home_button_start[0] <= x <= home_button_end[0] and home_button_start[1] <= y <= home_button_end[1]:
                    cap.release()
                    cv2.destroyAllWindows()
                    return render(request, 'homee.html')  # Render the home screen template

        else:
            prev_point = None

        # Overlay the canvas on the frame
        combined_frame = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
        cv2.imshow('Virtual Drawing with Colors & Home Button', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    return HttpResponse("<h1>Drawing Ended</h1>")





def login_view(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        
        # Authenticate the user
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('homee')  # Redirect to the homepage or dashboard
        else:
            messages.error(request, "Invalid username or password.")
            return render(request, 'homee.html', {'trigger_login': True})  # Trigger login modal

    
    return render(request, "login.html")  # Render login page with context


def logout_view(request):
    logout(request)
    return redirect('homee')  # Redirect to the homepage after logout


from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils.html import format_html

def sign_up_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])  # Ensure the form does not hash passwords already
            user.save()
            messages.success(request, format_html('<span class="text-blue-200">Account created successfully! You can now log in.</span>'))
            return render(request, 'homee.html', {'trigger_login': True})  # Trigger login modal
        else:
            messages.error(request, 'There was an error with your form submission: {}'.format(form.errors))
            return render(request, 'homee.html', {'trigger_signup': True})  # Trigger signup modal
    else:
        form = SignUpForm()

    return render(request, 'signup.html', {'form': form})



def members_list(request):
    members = Member.objects.all()
    return render(request, 'members/members_list.html', {'members': members})


def members(request):
    template = loader.get_template('homee.html')
    return HttpResponse(template.render())


def about(request):
    return render(request, 'about.html')




def homee(request):
    return render(request, 'homee.html')
