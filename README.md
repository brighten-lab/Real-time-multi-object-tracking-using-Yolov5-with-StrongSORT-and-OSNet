# Real-time multi-object tracking using Yolov5 with StrongSORT and OSNet

## 만든 사람들

[<img src="https://img.shields.io/badge/최재혁-181717?style=for-the-badge&logo=github&logoColor=white">](https://github.com/jjaegii)

- tracking/video record/image crop/measure distance 구현

[<img src="https://img.shields.io/badge/신혜민-181717?style=for-the-badge&logo=github&logoColor=white">](https://github.com/heymin2)

- 웹서버/siamese network 구현

[<img src="https://img.shields.io/badge/안수진-181717?style=for-the-badge&logo=github&logoColor=white">](https://github.com/ssuzyn)

- 로그인/회원가입/통계페이지 구현

## 소개

#### 기능 1

1. deep sort

- [Yolov5_StrongSORT_OSNet](https://github.com/mikel-brostrom/yolov8_tracking)을 활용하여 id 트래킹 구현

2. video record/save path to db

- detection 됐다면 녹화 시작, 최대 5분 녹화
- detection 된 것이 없다면 녹화 중단 후 동영상 저장 및 데이터베이스에 경로 저장

3. image crop/save path to db

- 녹화가 중지되는 시점에서 detection된 사람들의 box에서 위 3분의 1 지점까지 image crop하여 저장 및 데이터베이스에 경로와 동영상 경로 저장

#### 기능 2

1. measure distance

- tracking되는 사람의 이동 거리 값 측정

2. compare face

- Siamese Network를 활용하여 사람 얼굴을 비교 후 매칭

## 외부 설정

distance_measure.py의 fx, fy, h, theta_tilt, width,height를 사전 세팅해야함

1. distance_measure.py의 초점거리(fx, fy)를 구하는 방법

- cam_settings/pattern.png 프린트
- capture.py를 통하여 pattern.png 프린트 종이를 캡쳐
- camera_calibration.py를 통하여 초점 거리 구하여 적용
- 자세한 건 이동 거리 측정 프로젝트.pdf 파일 참고

2. theta_tilt 구하는 방법

- cam_settings/theta_tilt 구하는 방법.png를 참고하면 됨

나머지는 distance_measure.py에 주석 처리해놓은데로 구하면 됨

## 설치 및 실행

### 설치

```
git clone --recurse-submodules https://github.com/brighten-lab/kang_project.git yolo_tracking # clone recursively
cd yolo_tracking
pip install -r requirements.txt  # install dependencies
```

### 실행

```
# run.sh에서 --source 경로 수정, track.py에서 record 함수 내 cap 경로 수정 후 (둘다 동일한 경로여야함)
./run.sh
```

http://localhost:5000/ 로 접속하면 볼 수 있음

http://localhost:5001/ 에는 object tracking하는 화면만 따로 나옴

## Demo

### 로그인 페이지

![login](https://user-images.githubusercontent.com/77189999/217696833-78046ee1-b001-44c1-8a12-2b2ee2184434.png)

### 회원가입 페이지

![register](https://user-images.githubusercontent.com/77189999/217696835-e1662a09-92a4-40ba-b9ed-66b9a1b36d9a.png)

### 실시간 비디오 페이지

![video](https://user-images.githubusercontent.com/77189999/217696838-fabce760-12e3-4d88-9462-9386fcfa7c37.png)

### 가장 최근 녹화된 동영상 조회 페이지

![영상조회](https://user-images.githubusercontent.com/77189999/217696832-45c0b135-99f7-47f7-b4df-27ea4995120a.png)

### 통계 페이지

![통계](https://user-images.githubusercontent.com/77189999/217696828-ba827235-82c5-4a07-940a-b3278ebfa26a.png)
