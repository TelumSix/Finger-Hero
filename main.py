import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import random


# Hand landmark connections for drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm
]


class HandTracker:
    def __init__(self, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Create hand landmarker
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.timestamp = 0
    
    def find_hands(self, img, draw=True):
        """Find hands in the image and optionally draw landmarks."""
        # Convert BGR to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect hands
        self.results = self.detector.detect_for_video(mp_image, self.timestamp)
        self.timestamp += 1
        
        # Draw landmarks if requested
        if draw and self.results.hand_landmarks:
            h, w, c = img.shape
            
            for hand_landmarks in self.results.hand_landmarks:
                # Draw connections
                for connection in HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    
                    cv2.line(img, start_point, end_point, (0, 255, 0), 2)
                
                # Draw landmarks
                for landmark in hand_landmarks:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        """Get the position of hand landmarks."""
        landmark_list = []
        
        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                hand = self.results.hand_landmarks[hand_no]
                h, w, c = img.shape
                
                for id, landmark in enumerate(hand):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmark_list.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        
        return landmark_list
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two landmarks."""
        x1, y1 = p1[1], p1[2]
        x2, y2 = p2[1], p2[2]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance
    
    
    def get_finger_thumb_touches(self, landmark_list):
        """Detect which fingers are touching the thumb."""
        touches = {
            'Index': False,
            'Middle': False,
            'Ring': False,
            'Pinky': False
        }
        
        if len(landmark_list) == 0:
            return touches
        
        thumb_tip = landmark_list[4]
        
        # Check each finger
        index_tip = landmark_list[8]
        if self.calculate_distance(thumb_tip, index_tip) < 40:
            touches['Index'] = True
        
        middle_tip = landmark_list[12]
        if self.calculate_distance(thumb_tip, middle_tip) < 40:
            touches['Middle'] = True
        
        ring_tip = landmark_list[16]
        if self.calculate_distance(thumb_tip, ring_tip) < 40:
            touches['Ring'] = True
        
        pinky_tip = landmark_list[20]
        if self.calculate_distance(thumb_tip, pinky_tip) < 40:
            touches['Pinky'] = True
        
        return touches


class Note:
    """Represents a falling note in a lane."""
    def __init__(self, lane, start_y, speed=3, color=(255, 100, 0)):
        self.lane = lane  # 0: Index, 1: Middle, 2: Ring, 3: Pinky
        self.y = start_y
        self.speed = speed
        self.color = color
        self.active = True
        self.radius = 15
    
    def update(self):
        """Move the note down the lane."""
        self.y += self.speed
    
    def draw(self, img, lane_x, lane_width):
        """Draw the note on the image."""
        if self.active:
            center_x = lane_x + lane_width // 2
            cv2.circle(img, (center_x, int(self.y)), self.radius, self.color, -1)
            cv2.circle(img, (center_x, int(self.y)), self.radius, (255, 255, 255), 2)


class RhythmGame:
    """Manages the rhythm game logic."""
    def __init__(self, img_width, img_height):
        self.lanes = ['Index', 'Middle', 'Ring', 'Pinky']
        self.lane_count = 4
        self.lane_width = 80
        self.lane_spacing = 10
        self.start_x = 50
        
        self.img_height = img_height
        self.target_y = img_height - 100  # Where notes should be hit
        self.hit_tolerance = 40  # Pixels tolerance for hitting notes
        
        self.notes = []
        self.score = 0
        self.combo = 0
        self.max_combo = 0
        
        self.last_spawn_time = time.time()
        self.spawn_interval = 1.5  # Seconds between spawns
        
        self.colors = [
            (255, 100, 50),   # Index - Blue-ish
            (50, 255, 100),   # Middle - Green-ish
            (255, 200, 50),   # Ring - Yellow-ish
            (200, 50, 255)    # Pinky - Purple-ish
        ]
        
        self.last_touches = {finger: False for finger in self.lanes}
    
    def spawn_note(self):
        """Spawn a new note in a random lane."""
        lane = random.randint(0, self.lane_count - 1)
        note = Note(lane, -20, speed=4, color=self.colors[lane])
        self.notes.append(note)
    
    def update(self, touches):
        """Update game state."""
        current_time = time.time()
        
        # Spawn new notes
        if current_time - self.last_spawn_time > self.spawn_interval:
            self.spawn_note()
            self.last_spawn_time = current_time
        
        # Update existing notes
        notes_to_remove = []
        for note in self.notes:
            if not note.active:
                continue
            
            note.update()
            
            # Check if note passed the target (miss)
            if note.y > self.target_y + self.hit_tolerance + 20:
                notes_to_remove.append(note)
                self.combo = 0  # Reset combo on miss
        
        # Remove missed notes
        for note in notes_to_remove:
            self.notes.remove(note)
        
        # Check for hits (detect rising edge - finger just touched)
        for i, finger in enumerate(self.lanes):
            # Rising edge detection
            if touches[finger] and not self.last_touches[finger]:
                self.check_hit(i)
        
        # Update last touches
        self.last_touches = touches.copy()
    
    def check_hit(self, lane):
        """Check if a note was hit in the given lane."""
        for note in self.notes:
            if note.lane == lane and note.active:
                distance = abs(note.y - self.target_y)
                
                if distance <= self.hit_tolerance:
                    # Hit!
                    note.active = False
                    self.notes.remove(note)
                    
                    # Score based on accuracy
                    if distance <= 15:
                        points = 100  # Perfect
                        self.combo += 1
                    elif distance <= 30:
                        points = 50   # Good
                        self.combo += 1
                    else:
                        points = 25   # OK
                        self.combo += 1
                    
                    self.score += points * max(1, self.combo // 5)
                    self.max_combo = max(self.max_combo, self.combo)
                    return True
        return False
    
    def draw(self, img, touches):
        """Draw the game lanes, notes, and hit zones."""
        h, w, c = img.shape
        
        # Draw lanes
        for i in range(self.lane_count):
            lane_x = self.start_x + i * (self.lane_width + self.lane_spacing)
            
            # Lane background
            lane_color = (40, 40, 40)
            cv2.rectangle(img, 
                         (lane_x, 0), 
                         (lane_x + self.lane_width, h),
                         lane_color, -1)
            
            # Lane borders
            cv2.rectangle(img, 
                         (lane_x, 0), 
                         (lane_x + self.lane_width, h),
                         (80, 80, 80), 2)
            
            # Target zone at bottom
            target_zone_color = (60, 60, 60)
            cv2.rectangle(img,
                         (lane_x, self.target_y - self.hit_tolerance),
                         (lane_x + self.lane_width, self.target_y + self.hit_tolerance),
                         target_zone_color, -1)
            
            # Hit indicator (lights up when finger touches thumb)
            finger = self.lanes[i]
            if touches[finger]:
                indicator_color = self.colors[i]
                indicator_radius = 25
            else:
                indicator_color = (100, 100, 100)
                indicator_radius = 20
            
            center_x = lane_x + self.lane_width // 2
            cv2.circle(img, (center_x, self.target_y), indicator_radius, indicator_color, -1)
            cv2.circle(img, (center_x, self.target_y), indicator_radius, (255, 255, 255), 2)
            
            # Lane label
            label_y = h - 20
            cv2.putText(img, finger[0], (lane_x + 30, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)
        
        # Draw notes
        for note in self.notes:
            if note.active:
                lane_x = self.start_x + note.lane * (self.lane_width + self.lane_spacing)
                note.draw(img, lane_x, self.lane_width)
        
        # Draw score and combo
        score_x = self.start_x + self.lane_count * (self.lane_width + self.lane_spacing) + 20
        cv2.putText(img, f"Score: {self.score}", (score_x, 50),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        if self.combo > 0:
            combo_color = (0, 255, 255) if self.combo >= 10 else (255, 255, 255)
            cv2.putText(img, f"Combo: {self.combo}", (score_x, 90),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, combo_color, 2)
        
        cv2.putText(img, f"Best: {self.max_combo}", (score_x, 130),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 2)


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height
    
    # Initialize hand tracker
    tracker = HandTracker(detection_confidence=0.7, tracking_confidence=0.7)
    
    # Initialize rhythm game
    game = RhythmGame(1280, 720)
    
    prev_time = 0
    
    print("Hand Tracking Rhythm Game Started!")
    print("Touch your thumb to each finger to hit the notes!")
    print("Press 'q' to quit")
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            break
        
        # Flip image horizontally for mirror view
        img = cv2.flip(img, 1)
        
        # Get image dimensions
        h, w, c = img.shape
        
        # Find hands in the image
        img = tracker.find_hands(img, draw=False)  # Don't draw landmarks to keep clean UI
        landmark_list = tracker.find_position(img, draw=False)
        
        # Get finger touches
        touches = tracker.get_finger_thumb_touches(landmark_list)
        
        # Update and draw game
        game.update(touches)
        game.draw(img, touches)
        
        # Draw hand landmarks on the right side (smaller and less intrusive)
        if len(landmark_list) > 0:
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(landmark_list) and end_idx < len(landmark_list):
                    start_point = (landmark_list[start_idx][1], landmark_list[start_idx][2])
                    end_point = (landmark_list[end_idx][1], landmark_list[end_idx][2])
                    cv2.line(img, start_point, end_point, (0, 200, 0), 1)
            
            # Draw landmark points
            for landmark in landmark_list:
                cv2.circle(img, (landmark[1], landmark[2]), 3, (255, 0, 200), cv2.FILLED)
        
        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # FPS text
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(img, fps_text, (w - 120, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2)
        
        # Display the image
        cv2.imshow("Hand Tracking Rhythm Game - (Press q to quit)", img)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nGame Over!")
    print(f"Final Score: {game.score}")
    print(f"Max Combo: {game.max_combo}")


if __name__ == "__main__":
    main()
