# 다음의 모듈을 설치해주세요
# pip install requirements.txt
# pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
# pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# YOLO모델 import
from ultralytics import YOLO

# supervision 필요한 클래스, 함수 import
from supervision.video import VideoInfo, get_video_frames_generator, process_video
from supervision.detection.core import Detections
from supervision.detection.core import BoxAnnotator
from supervision.detection.polygon_zone import PolygonZone, PolygonZoneAnnotator
from supervision.draw.color import Color, ColorPalette
from supervision.geometry.core import Point
from supervision.detection.line_counter import LineZone, LineZoneAnnotator

# Tracking tools import
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from typing import List
import numpy as np

from dataclasses import dataclass

import os

# make tracks function
# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections, 
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)
    
    tracker_ids = [None] * len(detections)
    
    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

def make_result(polygon, TEST_VIDEO_PATH, CLASS_ID, LINE_START, LINE_END, model):
    
    # VideoInfo.from_video_path > VideoInfo(width, height, fps, total_frames) 반환
    video_info = VideoInfo.from_video_path(TEST_VIDEO_PATH) 
    
    
    # PolygonZone 객체 생성, video_info.resolution_wh > width, height 반환
    # 설정한 구역 안에 있는 사람 셀 때 사용 될 객체
    zone = PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
    
    
    # LineZone객체 생성 
    # 선을 기준으로 넘어가는 사람 셀 때 사용 될 객체
    line_counter = LineZone(start=LINE_START, end=LINE_END)
    
    # BYTETracker객체 생성 
    # https://bokonote.tistory.com/5
    byte_tracker = BYTETracker(BYTETrackerArgs())

    
    # initiate annotators
    # PolygonZoneAnnotator 객체 생성
    # 설정한 구역에 사람이 몇명인지 표시해줄 때 사용 될 객체
    zone_annotator = PolygonZoneAnnotator(zone=zone, color=Color.green(), thickness=2, text_thickness=1, text_scale=1, text_padding=3, text_color=Color.white())
    
    # LineZoneAnnotator 객체 생성
    # 선 기준으로 in out 사람이 몇명인지 표시해 줄 때 사용 될 객체
    line_annotator = LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=1, text_padding=3, text_offset=2.0, color = Color.black(), text_color=Color.white())
    
    
    def process_frame(frame: np.ndarray, _,CLASS_ID, model) -> np.ndarray:

        # detect
        # 모델을 통해 객체 탐지
        results = model(frame, imgsz=640)[0]
        detections = Detections.from_yolov8(results)
        detections = detections[detections.class_id == 0]
        
        # 사람이 몇명인지 count
        _, count = zone.trigger(detections=detections)

        # filtering out detections with unwanted classes
        # 원하는 클래스만 뽑아내기
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)
        
        # tracking detections
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)
        # filtering out detections without trackers
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # updating line counter
        # 승하차 사람 계수
        in_count, out_count = line_counter.trigger(detections=detections)
        
        # annotate and display frame
        # 승하차 사람 계수 한 것 화면에 표시
        line_annotator.annotate(frame=frame, line_counter=line_counter)
        
        # annotate
        # BoxAnnotator 객체 생성
        box_annotator = BoxAnnotator(thickness=1, text_thickness=1, text_scale=0, text_padding=0)
        
        # 객체 탐지 한것 표시
        frame = box_annotator.annotate(scene=frame, detections=detections)
        
        # 설정한 구역에 사람이 몇명인지 표시
        frame = zone_annotator.annotate(scene=frame)

        return frame, in_count, out_count, count
    
    # 이 함수에 대한 부분은 supervision.video.py에 있습니다
    process_video(source_path = TEST_VIDEO_PATH, save_folder = './result_video', target_path = f'./result_video/result.mp4', callback = process_frame, CLASS_ID = CLASS_ID, model = model)

    


########################이부분을 받아야됩니다.#########################
model = YOLO('l_best.pt') # 모델 경로 입력하기
TEST_VIDEO_PATH = './test_data/test3.mp4' # 비디오 파일 경로 입력
CLASS_ID = [0] # 머리만 detection
LINE_START = Point(288, 0) # 기준선 좌표 입력 
LINE_END = Point(370, 355)

polygon = np.array([[150, 0],[288, 0],[370, 355],[0, 359]])
# 영역 좌표 입력 numpy 배열로 입력할 것

make_result(polygon, TEST_VIDEO_PATH, CLASS_ID, LINE_START, LINE_END, model)