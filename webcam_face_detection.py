import cv2
from deepface import DeepFace
import time
from datetime import datetime

def format_gender(gender_dict):
    # gender_dict -- get highest vakue
    if not isinstance(gender_dict, dict):
        return str(gender_dict)
    dominant = max(gender_dict.items(), key=lambda kv: kv[1])[0]
    return dominant

print("🚀 Starting webcam emotion + age + gender detector...")

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
if not cap.isOpened():
    print("❌ ERROR: Cannot open webcam.")
    exit()

last_analysis_time = 0
analysis_interval = 2
last_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ ERROR: Cannot read frame from webcam.")
        break

    current_time = time.time()
    if current_time - last_analysis_time > analysis_interval:
        try:
            results = DeepFace.analyze(
                frame,
                actions=['emotion', 'age', 'gender'],
                enforce_detection=False
            )
            if not isinstance(results, list):
                results = [results]
            last_results = results
            last_analysis_time = current_time

            timestamp = datetime.now().strftime("%H:%M:%S")
            for idx, res in enumerate(results):
                person_id = idx + 1
                emotion = res.get('dominant_emotion', 'N/A')
                age = res.get('age', 'N/A')
                gender_raw = res.get('gender', {})
                gender = format_gender(gender_raw)
                age_str = int(age) if isinstance(age, (float, int)) else age
                print(f"[{timestamp}] Person {person_id}; Gender: {gender}; Emotion: {emotion}; Age: {age_str}")
        except Exception as e:
            print(f"⚠️ Analysis error: {e}")

    for idx, res in enumerate(last_results):
        region = res.get('region', {})
        x = int(region.get('x', 0))
        y = int(region.get('y', 0))
        w = int(region.get('w', 0))
        h = int(region.get('h', 0))
        emotion = res.get('dominant_emotion', 'N/A')
        age = res.get('age', 'N/A')
        gender_raw = res.get('gender', {})
        gender = format_gender(gender_raw)
        age_str = int(age) if isinstance(age, (float, int)) else age

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"P{idx+1}: {gender}, {emotion}, {age_str}"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("🎥 Live Multi-Face Emotion/Age/Gender Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting webcam.")
        break

cap.release()
cv2.destroyAllWindows()
