## Image Detection 방법론 
### RCNN, SPPnet, FastRCNN, FasterRCNN

<img src='https://www.dropbox.com/s/1o3tpj0erz1kafr/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2002%2057.png?raw=1'>
* Object Detection 문제 : Object가 여러개 있을 수 있는 문제(위치, classfication까지 정확하게)

<img src='https://www.dropbox.com/s/kv3yuo9kpsa8gxk/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2003%2040.png?raw=1'>

---

### 1. RCNN 계열

---

### 1) RCNN (Rich feature hierarchies for accurate object detection and semantic segmentation Tech report) [논문 링크](https://arxiv.org/pdf/1311.2524.pdf)

<img src='https://www.dropbox.com/s/czww7q4nzfm927h/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.05.32.png?raw=1'>

* 딥러닝을 이용한 Image Detection 시초 / 효과적으로 푼 논문
* 매우 간단한 방법론 = CNN이 잘되는 이유
	* 이미지 안에서 feature를 CNN으로 뽑아냄
	* 물체를 네모(이미지 영역)를 쳐줘야 하나, 쉽지 않은 문제
* Image Detection vs Segementic
	* segementic 생각하면 train data set이 굉장히 직관적으로 제공되어 end-to-end 학습이 쉬움
	* detection는 이미지가 있고, 이미지에 네모를 쳐줘야 함. end-to-end로 학습하기가 매우 어려움
	* 초창기 방법 : 이미지 안에서 바운딩 박스(네모 : 3천개 / 논문에서는 2천개)를 많이 뽑아서, (-> 기존에 있는 다른 방법론(딥러닝x) 적용) 이미지를 리사이즈(직사각형 -> 정사각형)하여 CNN에 집어 넣어 분류 진행
* 찾고자 하는 분류 개수가 20개면 21개로 설정 (1개는 백그라운드로 설정하여 고려하지 않음)

<img src='https://www.dropbox.com/s/mbw9aog42g7wty1/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.08.16.png?raw=1'>
<img src='https://www.dropbox.com/s/t7yaun01l1c7huw/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2006%2014.png?raw=1'>
<img src='https://www.dropbox.com/s/08utmxxxiq869rw/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2006%2054.png?raw=1'>

* components
	* 물체가 있을 것 같은 공간에 네모를 쳐주는 방법론 (딥러닝 x) / cpu로 하면 1개 이미지에 1분이 걸림
	* CNN 통과
	* 이미지의 피쳐를 SVM로 classification 

<img src='https://www.dropbox.com/s/fkw4t1et469c4ld/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.15.25.png?raw=1'>
<img src='https://www.dropbox.com/s/vtesuxty0q8egt7/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2008%2000.png?raw=1'>

* Region Proposals
	* 2개로 바운딩 박스를 잡았다 하더라도, 그 안에 내가 원하는 물체를 바운딩 박스하지 못한다면 무의미
	* merge : RGB, HSV, 영역들 크기, 합쳐졌을 때 hole이 생기는지? 고려해서 진행 

<img src='https://www.dropbox.com/s/ykxn7mod86wwkaj/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.16.37.png?raw=1'>

* Feature Extraction
	* Alexnet 사용, 옛날이기 때문에 googlenet 이나 resnet 이 없음
* 뒷 단에 있는 4,096(2,048 x 2) feature를 사용
* Test Time
	* 단점 : (오래 걸리는 이유) 2천개의 이미지가 CNN 돌아야 함 --> 보틀렉이 생김 (뒤에 나오는 논문들에서는 이 문제를 해결)
	* gpu - 13s/image , cpu - 53s/image

<img src='https://www.dropbox.com/s/4qa36927jgejfw3/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.17.48.png?raw=1'>

* Training
	* 학습 시킬때에는 시간을 줄일 방법이 없음
* Region Proposals를 통해 2천개를 뽑아, 뽑힌 네모가 실제 네모가 얼마나 겹치는지 확인 IoU >= 0.5면 Positive로 판단(실제 네모 o) , IoU <= 0.3이면 Negative로 판단(실제 네모 x), 아예 사용하지 않음

<img src='https://www.dropbox.com/s/trn09z7wjq0oi2q/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.18.56.png?raw=1'>

* Bounding box regression : Region Proposals로 찾은 네모를 어떻게 옮겨야 실제 네모와 같은지를 만드는 과정
* 하나의 네모를 다른 네모로 옮기기 위해서는 4개의 숫자가 필요 : 중심점, 너비, 높이, 종횡비

<img src='
https://www.dropbox.com/s/b5xr68n26kns7o9/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-18%2019.44.44.png?raw=1'>

* Result : 초창기 버전이라 성능이 높지는 않음
	* 너무 느림
	* SVM 과 regressor 결과를 CNN에 업데이트 시키지 못함
	* multi stage traing 필요


---

### 2) SPPnet (Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition) [논문 링크](https://arxiv.org/pdf/1406.4729.pdf)

* 내가 찾고 싶은 사물의 사이즈를 알 수가 없는 상태에서 그 사물을 찾음

<img src='https://www.dropbox.com/s/2qnf0slby27ed5a/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.28.52.png?raw=1'>

* RCNN에 가장 큰 단점을 보완 : bounding box 개수에 따라 CNN이 2,000번 돌았으나, CNN이 1번 돌음
* CNN은 fixed-input-size가 필요하여, RCNN은 여러 다른 사이즈를 갖는 bounding box를 추출함

<img src='https://www.dropbox.com/s/ktkntdi75eneigp/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.29.01.png?raw=1'>

* 전체 이미지를 CNN 통과 시켜서, CNN feature map 위에서 해당하는 영엑에 있는 어떤 정보를 빼옴 

<img src='https://www.dropbox.com/s/dqkmojefsnnrve8/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-18%2020.02.44.png?raw=1'>

* CNN feature map에 어떤 영역을 resize하지 않고, spatial pyramid polling 하여 처리
* 예 : 2 x 5 에서 10개의 숫자를 평균내서 5개의 숫자를 구함
* 예 : 4 x 2 에서 8개의 숫자를 평균내서 5개의 숫자를 구함

<img src='https://www.dropbox.com/s/q65axvhlcsd9fom/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.31.40.png?raw=1'>

* Result : RCNN에 비해 더 큰 영역에서 성과를 보여, 성능이 높아진 것을 확인

---

### 3) Fast R-CNN [논문 링크](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)

<img src='https://www.dropbox.com/s/zlw5md87erhqtkb/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.32.16.png?raw=1'>

* RCNN, SPPnet과의 차이점
	* RCNN -> SPPnet : CNN 1번 실행
	* SPPnet -> Fast RCNN : 어떤 Some region을 추출
	* Fast RCNN과 SPPnet의 차이가 거의 없음
* Fix-length represent를 찾는 방법
	* Fast RCNN : RoI Pooling
	* SPPnet : spatial pyramid polling

<img src='https://www.dropbox.com/s/w92u9fgw419ln5z/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.33.27.png?raw=1'>
<img src='https://www.dropbox.com/s/hhwbldqmed3tvzw/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.34.01.png?raw=1'>
<img src='https://www.dropbox.com/s/djnx24t2o0ska1d/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2017%2000.png?raw=1'>

* CNN이 돌아서 CNN feature map 추출
* RoI Projection를 통해 이미지에서 얻은 RoI 영역을 feature map 위에 얹어서 그 영역을 crop 함
* region 크기가 다르기 때문에 서로 다른 사이즈에 아웃풋이 나와서, RoI pooling를 통하여 Fixed Size feature vector로 추출
	* Bounding Box에 대한 classififer
	* Bounding Box를 어디로 옮길지에 대한 Bounding Box Regression
* Loss : classfifer와 regression 2개를 지님

<img src='https://www.dropbox.com/s/ti262nedf0icnqu/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.35.12.png?raw=1'>
<img src='https://www.dropbox.com/s/dataigl5xld1o5u/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2017%2008.png?raw=1'>
<img src='https://www.dropbox.com/s/5yyxkq7x41tiv7d/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2017%2013.png?raw=1'>

* SPP와 RoI Pooling 차이
	* SPP : 여러개의 bounding box를 합침
	* RoI Pooling : 그냥 미리 정해놓은 사이즈 만큼 잘라, 여기에 해당하는 값을 평균내서 9개의 vector를 만듬
* Result
	* test-time에 bottlenect 발생 -> region proposal이 밖에서 돌기 때문에(2.3초)
 
---

### 4) Faster R-CNN [논문 링크](https://arxiv.org/pdf/1506.01497.pdf)

<img src='https://www.dropbox.com/s/zl6tu1oh6uho78j/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.36.47.png?raw=1'>
<img src='https://www.dropbox.com/s/0hgjff1ag6grjs8/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2022%2001.png?raw=1'>
<img src='https://www.dropbox.com/s/3dkhfx52tb5my7a/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2025%2048.png?raw=1'>

* RCNN : 엄청 라이브하고, CNN이 여러번 돌아서 오래 걸림
* SPPnet : 성능이 좋지 않으며, multi stage traing이 됨
* Fast RCNN : bounding box sampling 해야 하기 때문에 느림
* Faster RCNN : bounding box sampling 조차 CNN 해보자! -> 그 bounding box를 Fast RCNN에 집어 넣어서 해보자!

<img src='https://www.dropbox.com/s/iv0lthkcgg7ijks/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.37.34.png?raw=1'>

* Pyramids of images : 이미지 자체의 사이즈를 줄여서, 줄여든 여러개의 이미지를(multiple scale image) 한번에 고려하는 것
* Pyramids of filters : 필터 사이즈를 바꿔서 적용
* Pyramids of anchors : 미리 정해져 있는 어떤 bounding box 사이즈를 얼마나 조절해야 원래 이미지 사이즈를 찾을 수 있을지 탐색 (9개 anchors를 사용)

<img src='https://www.dropbox.com/s/bunz532o459uxp6/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.38.44.png?raw=1'>
<img src='https://www.dropbox.com/s/nb7utgqhpo0fcpe/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2027%2018.png?raw=1'>
<img src='https://www.dropbox.com/s/d5cdr5siceurov2/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2028%2032.png?raw=1'>

* Region Proposal Network (새로운 파트) : bounding box만 output
* classfi : bounding box에 물체가 있는지 없는지? / regress box locations : 바운딩 박스에 위치 조정

<img src='https://www.dropbox.com/s/ov2dndu9a95jwc1/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-16%2019.39.48.png?raw=1'>
<img src='https://www.dropbox.com/s/aoonbcp2aq37wxm/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2029%2001.png?raw=1'>

* Region Proposal Network
	* 하나의 이미지가 들어 왔을 때 CNN을 돌리고 나면, 어떤 CNN feature map이 나옴
	* 원래 이미지가 100 x 100 일 경우 -> CNN feature map 은 32 x 32 x 512 로 나왔다고 할 경우
	* 총 1024개의(32 x 32) 픽셀들이 있음
	* feature map마다 k(9)개의 anchors box를 정의하고, 그 중에서 "어떤 anchors box가 좋은지? 어디로 옮겨야 할 지? (2가지)" 를 학습
	* 9개의 anchors 마다 2개 값을 할당 (postive, negative)해서 쓸만한 anchors를 판별 
	* IoU >= 0.5면 Positive로 판단(실제 네모 o) , IoU <= 0.3이면 Negative로 판단(실제 네모 x), 아예 사용하지 않음

<img src='https://www.dropbox.com/s/ntyjfdr9r6n7vt7/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-19%2013.22.19.png?raw=1'>

* region proposal network 는 k[anchors] x (4[bounding box reg]+2[bounding box ds]) vector를 뽑아줌
* FCN과 똑같음
* 우리가 물리적으로 원하는 어떤 현상이 있고, 그 현상을 딥러닝으로 구현하기 위해선, 어떤 CNN feature map에서 다른 CNN feature map으로 가는 mapping 밖에 못 찾으나, 그 mapping에 의미를 찾음 (이해가 안됨)

<img src='https://www.dropbox.com/s/ux9kc8uav8rnu3f/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-19%2013.23.44.png?raw=1'>

* Result : 성능이 좋음, 큰 물체와 작은 물체 모두 잘 잡음
	* 7에 배수가 아닌 값은 버림을 할 수 밖에 없는데, Boundig Box의 좌표 / 정확도가 필요할 때에 오차가 발생

<img src='https://www.dropbox.com/s/6upy7uwso2x74vw/%EC%82%AC%EC%A7%84%202018.%2011.%2014.%20%EC%98%A4%EC%A0%84%209%2004%2051.png?raw=1'>

* Faster RCNN : 성능이 가장 높음
* R-FCN / SSD가 속도는 빠른 편


---

### 정리

* RCNN
<img src='https://www.dropbox.com/s/u6xhbk1ue8jhzd7/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-19%2013.25.32.png?raw=1'>

* SPPnet
<img src='https://www.dropbox.com/s/w1f74797wr33pin/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-19%2013.26.07.png?raw=1'>

* Fast RCNN
<img src='https://www.dropbox.com/s/y2dqbt08mvxzwzd/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-19%2013.26.10.png?raw=1'>

* Faster RCNN
<img src='https://www.dropbox.com/s/lmxdvft6g0sfkff/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-19%2013.26.47.png?raw=1'>

* 전체
<img src='https://www.dropbox.com/s/mkwkh6prv192gso/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202018-12-19%2013.26.53.png?raw=1'>


---

### References
* edwith 논문으로 짚어보는 딥러닝의 맥 - [Image Detection 방법론](https://www.edwith.org/deeplearningchoi/lecture/15568/)
* PR-012 Faster R-CNN - [Faster R-CNN](https://www.youtube.com/watch?v=kcPAGIgBGRs&list=PLWKf9beHi3Tg50UoyTe6rIm20sVQOH1br&index=13)




















