# pica_model
***

초기에 다음의 모듈을 설치해주세요.
```cmd
pip install - requirements.txt
pip install -e git+https://github.com/samson-wang/cython_bbox.git#egg=cython-bbox
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
***
모델을 실행하기 지정해줘야 할 변수 예시
```python
# 파일 하나를 넣는 경우
model = YOLO('l_best.pt') # 모델 경로 입력하기
TEST_VIDEO_PATH = './test_data/test3.mp4' # 비디오 파일 경로 입력
CLASS_ID = [0] # 머리만 detection
LINE_START = Point(288, 0) # 기준선 좌표 입력 
LINE_END = Point(370, 355)
polygon = np.array([[150, 0],[288, 0],[370, 355],[0, 359]])

# 파일 여러개를 넣는경우
model = YOLO('l_best.pt') # 모델 경로 입력하기
TEST_VIDEO_FOLDER_PATH = './test_data/' # 비디오 폴더 경로 입력
CLASS_ID = [0] # 머리만 detection
LINE_START = [Point(0,720), Point(350, 360), Point(288, 0), Point(785, 0), Point(720, 250), Point(720, 250), Point(720, 250), Point(720, 250)] # 기준선 좌표 입력 [동영상1, 동영상2, ...]
LINE_END = [Point(260,80), Point(260, 0), Point(370, 355), Point(1000, 718),  Point(350,1080),  Point(350,1080),  Point(350,1080),  Point(350,1080)]

polygon = [np.array([[260,80], [0,720], [406,720], [406,180]]),
           np.array([[350,360], [260,0], [0,0], [0,360]]),
           np.array([[150, 0],[288, 0],[370, 355],[0, 359]]),
           np.array([[0,0],[0,718], [1000, 718], [785,0]]),
           np.array([[720,250],[350,1080], [0, 1080], [0,200],[700, 200]]),
           np.array([[720,250],[350,1080], [0, 1080], [0,200],[700, 200]]),
           np.array([[720,250],[350,1080], [0, 1080], [0,200],[700, 200]]),
           np.array([[720,250],[350,1080], [0, 1080], [0,200],[700, 200]])
           ]
# 영역 좌표 입력 [동영상1, 동영상2, ...] numpy 배열로 입력할 것

make_result(polygon, TEST_VIDEO_FOLDER_PATH, CLASS_ID, LINE_START, LINE_END, model)
```


***
파일 하나를 넣는 경우 cmd창에 다음 코드를 입력하세요.
```cmd
python pica_file.py
```
***
파일 폴더를 넣는 경우 cmd창에 다음 코드를 입력하세요.
```cmd
python pica_folder.py
```
