# app.py
import cv2
import av
import mediapipe as mp
import numpy as np
import threading
import time
from pathlib import Path
from datetime import datetime
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ---------------- 설정 ----------------
CAPTURE_DIR = Path("captures")
CAPTURE_DIR.mkdir(exist_ok=True)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# 전역 캡처 저장소 (thread-safe)
LATEST_CAPTURE = {"bytes": None, "fname": None, "ts": None}
CAP_LOCK = threading.Lock()

# ---------------- Shaka 판별 함수 ----------------
def is_shaka(hand, w, h):
    """MediaPipe hand landmarks 기반 샤카 판별 (thumb and pinky up, others down)."""
    def c(i):
        lm = hand.landmark[i]
        return int(lm.x * w), int(lm.y * h)

    thumb_tip = c(4); thumb_ip = c(3)
    index_tip = c(8); index_kn = c(5)
    middle_tip = c(12); middle_kn = c(9)
    ring_tip = c(16); ring_kn = c(13)
    pinky_tip = c(20); pinky_kn = c(17)

    thumb_up = thumb_tip[1] < thumb_ip[1]         # 엄지 펴짐
    pinky_up = pinky_tip[1] < pinky_kn[1]         # 새끼 펴짐

    index_down  = index_tip[1] > index_kn[1]
    middle_down = middle_tip[1] > middle_kn[1]
    ring_down   = ring_tip[1] > ring_kn[1]

    return thumb_up and pinky_up and index_down and middle_down and ring_down

# ---------------- VideoProcessor ----------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # 각 worker/스레드 별로 Mediapipe 객체를 생성
        self.mp_face = mp.solutions.face_detection
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.face_detector = self.mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)
        self.hand_detector = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6)
        self.captured = False
        self.last_capture_time = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Mediapipe 처리
        face_res = self.face_detector.process(rgb)
        hand_res = self.hand_detector.process(rgb)

        face_detected = face_res.detections is not None
        shaka_detected = False

        # 손 처리: 샤카 체크
        if hand_res.multi_hand_landmarks:
            for hand_landmarks in hand_res.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                if is_shaka(hand_landmarks, w, h):
                    shaka_detected = True
                    # 중앙 텍스트 (간단 표시)
                    cv2.putText(img, "Shaka!", (w//2 - 140, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), 6)
                    break

        # 얼굴 박스 그리기
        if face_detected:
            for d in face_res.detections:
                self.mp_draw.draw_detection(img, d)

        # 캡처 조건: 얼굴 + 샤카 + (디바운스: 1초)
        if face_detected and shaka_detected and not self.captured:
            now = time.time()
            # 간단 디바운스: 마지막 캡처로부터 1초 이상 지나야 허용
            if now - self.last_capture_time > 1.0:
                fname = CAPTURE_DIR / f"shaka_{int(now)}.jpg"
                # 저장(서버 내)
                cv2.imwrite(str(fname), img)
                self.last_capture_time = now
                self.captured = True

                # 모듈 전역 변수에 JPEG bytes 저장 (thread-safe)
                _, jpg = cv2.imencode('.jpg', img)
                with CAP_LOCK:
                    LATEST_CAPTURE["bytes"] = jpg.tobytes()
                    LATEST_CAPTURE["fname"] = str(fname)
                    LATEST_CAPTURE["ts"] = datetime.fromtimestamp(now).isoformat(timespec='seconds')

        # 리셋: 샤카가 안 보이면 재촬영 가능
        if not shaka_detected:
            self.captured = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Shaka Shot", layout="centered")
st.title("🤙 Shaka Shot — 자동 촬영 앱 (Streamlit + streamlit-webrtc)")

col1, col2 = st.columns([3,1])

with col1:
    st.markdown("**카메라 스트림** — 브라우저에서 카메라 권한 허용 필요")
    ctx = webrtc_streamer(
        key="shaka-shot",
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col2:
    st.markdown("**설정 / 캡처**")
    st.write("- 얼굴 감지 + Shaka(엄지+새끼 펴짐) 인식 시 자동 캡처")
    st.write("- 캡처 파일은 서버의 `captures/` 폴더에 저장")
    st.write("- 'Refresh' 버튼으로 최신 캡처 확인")
    st.write("- 'Download' 으로 파일 저장")

    if st.button("Refresh latest capture"):
        with CAP_LOCK:
            if LATEST_CAPTURE["bytes"] is not None:
                st.image(LATEST_CAPTURE["bytes"], caption=f"Latest: {LATEST_CAPTURE['fname']} ({LATEST_CAPTURE['ts']})")
                st.download_button("Download latest", data=LATEST_CAPTURE["bytes"], file_name=Path(LATEST_CAPTURE["fname"]).name, mime="image/jpeg")
            else:
                st.info("아직 캡처된 이미지가 없습니다.")

    st.write("---")
    st.write("개발자 메모:")
    st.write(" - 포즈 감지 민감도는 조명/카메라 각도에 따라 달라질 수 있음.")
    st.write(" - 필요한 확장: 좌/우 손 구분, 카운트다운, 오디오 알림 등")

# 자동으로 새 캡처가 들어왔는지 UI에서 주기적으로 확인하고 싶다면,
# streamlit.experimental_set_query_params / st_autorefresh 등을 활용해서 자동 새로고침 추가 가능.
