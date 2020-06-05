import tkinter as tk

from PIL import Image, ImageTk

import cv2 as cv
from app.ctx import (age_gender_recognition_net, emotion_recognition_net,
                     face_detection_net, face_reidentification_net,
                     head_pose_estimation_net, landmark_detection_net,
                     mobilenet_ssd, person_detection_net,
                     person_reidentification_net)
from app.FeatureDataBase import FeatureDataBase
from app.Stream import Stream


class App:
    def __init__(self, source=0):
        self.stream = Stream(source)
        self.person_db = FeatureDataBase()
        self.face_db = FeatureDataBase()
        self.face_position_buffer = {}
        self.setup_gui()

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Openvino Workshop")

        # Checkbox values
        self.is_face_detection = tk.BooleanVar()
        self.is_landmark_detection = tk.BooleanVar()
        self.is_face_reidentification = tk.BooleanVar()
        self.is_age_gender_recognition = tk.BooleanVar()
        self.is_emotion_recognition = tk.BooleanVar()
        self.is_head_pose_estimation = tk.BooleanVar()
        self.is_person_detection = tk.BooleanVar()
        self.is_person_reidentification = tk.BooleanVar()

        # GUI elements
        self.stream_view = tk.Label(self.root, bd=0)
        self.stream_fps = tk.Label(self.root, bd=0)
        self.stream_runtime = tk.Label(self.root, bd=0)

        face_detection_checkbutton = tk.Checkbutton(
            text="Face Detection",
            variable=self.is_face_detection,
        )

        landmark_detection_checkbutton = tk.Checkbutton(
            text="Landmark Detection",
            variable=self.is_landmark_detection,
        )

        face_reidentification_checkbutton = tk.Checkbutton(
            text="Face Reidentification",
            variable=self.is_face_reidentification,
        )

        age_gender_recognition_checkbutton = tk.Checkbutton(
            text="Age Gender Recognition",
            variable=self.is_age_gender_recognition,
        )

        emotion_recognition_checkbutton = tk.Checkbutton(
            text="Emotion Recognition",
            variable=self.is_emotion_recognition,
        )

        head_pose_estimation_checkbutton = tk.Checkbutton(
            text="Head Pose Estimation",
            variable=self.is_head_pose_estimation,
        )

        person_detection_checkbutton = tk.Checkbutton(
            text="Person Detection",
            variable=self.is_person_detection,
        )

        person_reidentification_checkbutton = tk.Checkbutton(
            text="Person Reidentification",
            variable=self.is_person_reidentification,
        )

        # GUI element positions
        self.stream_view.grid(row=0, column=0, rowspan=100, columnspan=1)
        self.stream_fps.grid(row=98, column=1, rowspan=1, columnspan=1)
        self.stream_runtime.grid(row=99, column=1, rowspan=1, columnspan=1)
        face_detection_checkbutton.grid(row=0, column=1, sticky=tk.W)
        landmark_detection_checkbutton.grid(row=1, column=1, sticky=tk.W)
        face_reidentification_checkbutton.grid(row=2, column=1, sticky=tk.W)
        age_gender_recognition_checkbutton.grid(row=3, column=1, sticky=tk.W)
        emotion_recognition_checkbutton.grid(row=4, column=1, sticky=tk.W)
        head_pose_estimation_checkbutton.grid(row=5, column=1, sticky=tk.W)
        person_detection_checkbutton.grid(row=6, column=1, sticky=tk.W)
        person_reidentification_checkbutton.grid(row=7, column=1, sticky=tk.W)

    def video_loop(self):
        frame = self.stream.get_frame()
        output_image = frame.copy()

        if self.is_person_detection.get():
            persons = mobilenet_ssd.detect(frame, 0.75)
            mobilenet_ssd.draw(output_image)

            if self.is_person_reidentification.get():
                for pxmin, pymin, pxmax, pymax in persons:
                    person_crop = frame[pymin:pymax, pxmin:pxmax]

                    person_descriptor = person_reidentification_net.get_descriptor(
                        person_crop)

                    person_id = self.person_db.get_id(person_descriptor, 0.65)

                    if not person_id:
                        self.person_db.add(person_descriptor)

                    scale = ((pxmax - pxmin) / 150)

                    if scale < 0.5:
                        scale = 0.5

                    elif scale > 1:
                        scale = 1

                    weight = int((pxmax - pxmin) / 50)

                    if weight > 2:
                        weight = 2

                    cv.putText(output_image, f"id:{person_id}", (pxmin, pymin - 5),
                               cv.FONT_HERSHEY_COMPLEX, scale, (0, 255, 0), weight)

        if self.is_face_detection.get():
            faces = face_detection_net.detect(frame, 0.6)
            face_detection_net.draw(output_image)
            face_temp_dict = {}

            for fxmin, fymin, fxmax, fymax in faces:
                face_region = (fxmin, fymin, fxmax, fymax)
                face_center = (fxmin + (fxmax-fxmin)/2,
                               fymin + (fymax-fymin)/2)
                face_crop = frame[fymin:fymax, fxmin:fxmax]

                if self.is_landmark_detection.get():
                    landmark_detection_net.detect(face_crop, (fxmin, fymin))
                    landmark_detection_net.draw(output_image)

                if self.is_face_reidentification.get():
                    face_descriptor = face_reidentification_net.get_descriptor(
                        face_crop)

                    face_id = self.face_db.get_id(face_descriptor, 0.75)

                    if not face_id:
                        self.face_db.add(face_descriptor)

                    if face_id:
                        face_temp_dict[face_id] = face_center

                    min_d = (fxmax - fxmin)
                    nearest_id = 0

                    for ID in self.face_position_buffer:
                        dot = self.face_position_buffer[ID]

                        d = ((face_center[0] - dot[0]) ** 2 +
                             (face_center[1] - dot[1]) ** 2) ** 0.5

                        if d < min_d:
                            min_d = d
                            nearest_id = ID

                    if (nearest_id == face_id != 0):
                        self.face_db.update(face_id, face_descriptor)
                        self.face_position_buffer[face_id] = face_center

                    scale = ((fxmax - fxmin) / 150)

                    if scale < 0.5:
                        scale = 0.7

                    elif scale > 1:
                        scale = 1

                    weight = int((fxmax - fxmin) / 50)

                    if weight > 2:
                        weight = 2

                    cv.putText(output_image, f"id:{face_id}", (fxmin, fymin - 5),
                               cv.FONT_HERSHEY_COMPLEX, scale, (0, 255, 0), weight)

                if self.is_age_gender_recognition.get():
                    age_gender_recognition_net.detect(face_crop)
                    age_gender_recognition_net.draw(output_image, face_region)

                if self.is_emotion_recognition.get():
                    emotion_recognition_net.detect(face_crop)
                    emotion_recognition_net.draw(output_image, face_region)

                if self.is_head_pose_estimation.get():
                    head_pose_estimation_net.detect(face_crop)
                    head_pose_estimation_net.draw(output_image, face_region)

            self.face_position_buffer = face_temp_dict

        self.show_image(output_image)
        self.stream_fps.configure(
            text=f"Fps: {self.stream.fps:.1f}")
        self.stream_runtime.configure(
            text=f"Runtime: {self.stream.runtime:.0f}s")
        self.root.after(17, self.video_loop)

    def show_image(self, output_image):
        output_image = cv.cvtColor(output_image, cv.COLOR_BGR2RGBA)
        output_image = Image.fromarray(output_image)
        output_image = ImageTk.PhotoImage(output_image)

        self.stream_view.configure(image=output_image)
        self.stream_view.image = output_image

    def run(self):
        self.video_loop()
        self.root.mainloop()
