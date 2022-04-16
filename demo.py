#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils.cvfpscalc import CvFpsCalc
from utils.cvoverlayimg import CvOverlayImage


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--eye", type=str, default='image/eye03.png')
    parser.add_argument(
        "--eye_select",
        type=int,
        default=0,
        choices=[0, 1, 2],  # 0:Each Eye 1:Left Eye 2:Right Eye
    )

    parser.add_argument(
        "--unuse_mirror",
        action="store_true",
    )

    parser.add_argument("--max_num_faces", type=int, default=1)
    parser.add_argument(
        "--min_detection_confidence",
        help='min_detection_confidence',
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help='min_tracking_confidence',
        type=int,
        default=0.5,
    )
    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    eye_image_path = args.eye
    eye_select = args.eye_select

    max_num_faces = args.max_num_faces
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    unuse_mirror = args.unuse_mirror

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # 写輪眼画像読み込み
    eye_image = cv.imread(eye_image_path, cv.IMREAD_UNCHANGED)

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        if not unuse_mirror:
            image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # 描画 ################################################################
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # 虹彩の外接円の計算
                left_eye, right_eye = None, None
                left_eye, right_eye = calc_iris_min_enc_losingCircle(
                    debug_image,
                    face_landmarks,
                )

                # 描画
                debug_image = draw_landmarks(
                    debug_image,
                    face_landmarks,
                    left_eye,
                    right_eye,
                    eye_image,
                    eye_select,
                    unuse_mirror,
                )

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('NARUTO Sharingan Iris Overlay ', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_bounding_rect(points):
    landmark_array = np.empty((0, 2), int)
    for _, point in enumerate(points):
        landmark_point = [np.array((point[0], point[1]))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_iris_min_enc_losingCircle(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    visibility_presence = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append((landmark_x, landmark_y))
        visibility_presence.append((landmark.visibility, landmark.presence))

    left_eye_points = [
        landmark_point[468],
        landmark_point[469],
        landmark_point[470],
        landmark_point[471],
        landmark_point[472],
    ]
    right_eye_points = [
        landmark_point[473],
        landmark_point[474],
        landmark_point[475],
        landmark_point[476],
        landmark_point[477],
    ]

    left_eye_info = calc_min_enc_losingCircle(left_eye_points)
    right_eye_info = calc_min_enc_losingCircle(right_eye_points)

    return left_eye_info, right_eye_info


def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)

    return center, radius


def draw_landmarks(
    image,
    landmarks,
    left_eye,
    right_eye,
    eye_image,
    eye_select,
    unuse_mirror,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append((landmark_x, landmark_y))

    if len(landmark_point) > 0:
        # 左目マスク生成
        left_eye_outline = []
        left_eye_outline.append(landmark_point[133])
        left_eye_outline.append(landmark_point[173])
        left_eye_outline.append(landmark_point[157])
        left_eye_outline.append(landmark_point[158])
        left_eye_outline.append(landmark_point[159])
        left_eye_outline.append(landmark_point[160])
        left_eye_outline.append(landmark_point[161])
        left_eye_outline.append(landmark_point[246])
        left_eye_outline.append(landmark_point[144])
        left_eye_outline.append(landmark_point[145])
        left_eye_outline.append(landmark_point[153])
        left_eye_outline.append(landmark_point[154])
        left_eye_outline.append(landmark_point[155])
        left_eye_outline.append(landmark_point[133])
        left_mask = np.zeros((image_height, image_width, 3)).astype(np.uint8)
        cv.fillPoly(left_mask, [np.array(left_eye_outline)], (255, 255, 255))

        # 右目マスク生成
        right_eye_outline = []
        right_eye_outline.append(landmark_point[362])
        right_eye_outline.append(landmark_point[398])
        right_eye_outline.append(landmark_point[384])
        right_eye_outline.append(landmark_point[385])
        right_eye_outline.append(landmark_point[386])
        right_eye_outline.append(landmark_point[387])
        right_eye_outline.append(landmark_point[388])
        right_eye_outline.append(landmark_point[466])
        right_eye_outline.append(landmark_point[390])
        right_eye_outline.append(landmark_point[373])
        right_eye_outline.append(landmark_point[374])
        right_eye_outline.append(landmark_point[380])
        right_eye_outline.append(landmark_point[381])
        right_eye_outline.append(landmark_point[382])
        right_mask = np.zeros((image_height, image_width, 3)).astype(np.uint8)
        cv.fillPoly(right_mask, [np.array(right_eye_outline)], (255, 255, 255))

        # 左目重畳表示
        if eye_select == 0 or \
            (eye_select == 1 and not unuse_mirror) or \
                (eye_select == 2 and unuse_mirror):
            overlay_left_eye_image = cv.resize(
                eye_image,
                (int(left_eye[1] / 2) * 4, int(left_eye[1] / 2) * 4))
            temp_image = CvOverlayImage.overlay(
                image, overlay_left_eye_image,
                (left_eye[0][0] - left_eye[1], left_eye[0][1] - left_eye[1]))

            image = np.where(left_mask == 0, image, temp_image)

        # 右目重畳表示
        if eye_select == 0 or \
            (eye_select == 2 and not unuse_mirror) or \
                (eye_select == 1 and unuse_mirror):
            overlay_right_eye_image = cv.resize(
                eye_image,
                (int(right_eye[1] / 2) * 4, int(right_eye[1] / 2) * 4))
            temp_image = CvOverlayImage.overlay(
                image, overlay_right_eye_image,
                (right_eye[0][0] - right_eye[1],
                 right_eye[0][1] - right_eye[1]))

            image = np.where(right_mask == 0, image, temp_image)

    return image


if __name__ == '__main__':
    main()
