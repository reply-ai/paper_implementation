## RNN을 이용해 손글씨를 만드는 Handwriting generation

<img src='https://www.dropbox.com/s/jmlpt96akd8ny5p/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2020.55.13.png?raw=1'>

## 1. Structures of Recurrent Neural Networks(RNNs)
* Feed forward Neural Network
<img src='https://www.dropbox.com/s/g8ql6khef022tf3/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2020.58.45.png?raw=1'>
	* 순차적으로 입력이 있으면 출력이 있음

* RNN
<img src='https://www.dropbox.com/s/yhgsoyl67g4x22e/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2021.05.03.png?raw=1'>
<img src='https://www.dropbox.com/s/2s6pwidtelqmb35/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2021.05.34.png?raw=1'>
	* 이전에 있던 어떤 값이 현재에 다시 들어옴
	* `x` : input
	* `h` : hidden layer
	* `y` : output
	* `t` : time-step
 
* Deep Recurrent Neural Network
<img src='https://www.dropbox.com/s/vta7tz2lh1995vc/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2021.16.15.png?raw=1'>
	* 좀 더 복잡한 뉴럴 네트워크를 중간중간에 끼워넣음 (전체 구조가 변하지 않음)

* Stacked Recurrent Neural Network
<img src='https://www.dropbox.com/s/we9ob7z9bspey5r/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2021.18.10.png?raw=1'>
	* `g_t` : 히든 레이어 위에 하나의 레이어를 더 쌓음
	* 하단의 `h_t(cell)`는 short term depends(단기 의존성) 를 잡고, 상단의 `g_t(cell)`는 long term depends(장기 의존성)를 잡음
	* 참고 - [RNN : 장기 종속성 문제](https://brunch.co.kr/@chris-song/9) : 문장일 경우, 여러 문장 표현의 순서상 갭이 커질수록, RNN은 두 정보의 문맥을 연결하기 힘듬
 
## 2. Back Propagation through Time

* Training
<img src='https://www.dropbox.com/s/14gnlkrdmafl5v2/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2022.57.25.png?raw=1'>
<img src='https://www.dropbox.com/s/sluljgqbjbk8ktr/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2022.58.34.png?raw=1'>
<img src='https://www.dropbox.com/s/2jtz6b9kw8arqvs/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.00.30.png?raw=1'>
	* `x_t`를 긴 입력으로 보고 `y_t`를 긴 출력으로 보고 하나의 구조로 볼 수 있다. = parameters shared = back propagation through time
	* Exploding Gradient : 어떤 값이 1보다 크며, 계속 반복해서 곱해진 경우
	* Vanishing Gradient : 어떤 값이 1보다 작으며, 계속 반복해서 곱해진 경우

* Unfolding in Time : 시간 순으로 푸는 것
<img src='https://www.dropbox.com/s/yw0zmhlrnz753t9/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.01.48.png?raw=1'>
<img src='https://www.dropbox.com/s/0chc7pxgagfnh75/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.01.26.png?raw=1'>
	* 풀어서, feed foward neural network로 학습 시키는 것 : Back-Propagation Through Time

* Simple RNN
<img src='https://www.dropbox.com/s/1lhpgi4is665k7s/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.03.40.png?raw=1'>

* GRU
<img src='https://www.dropbox.com/s/7q5p7fucquhjx3e/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.03.44.png?raw=1'>
	* LSTM과 차이
		* LSTM : input, forget, output gate
		* GRU : reset, dynamic gate
		* reset : 이전 cell state를 얼마나 업데이트 할지 결정
		* dynamic : 현재 가지고 있는 candidate 얼마나 반영하고 싶을지 결정

* LSTM
<img src='https://www.dropbox.com/s/l7cpuwz4yl5sbyn/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.03.53.png?raw=1'>
<img src='https://www.dropbox.com/s/o3pbii1f44u55fb/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.04.03.png?raw=1'>
	* gate : sigmoid
	* tanh : cell state update, output = 값을 나타낼때

## 3. Generating Sequences from RNNs
<img src='https://www.dropbox.com/s/4sh4inwfoygxfw9/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.04.06.png?raw=1'>

* Generating Sequences from RNNs
<img src='https://www.dropbox.com/s/rsshko24rvliyk3/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.04.18.png?raw=1'>
	* 무엇가를 만드는데 집중
	* 이전 입력들이 주어지면, 다음번 입력들에 대한 확률 분포를 생성

* The Role of Memory : 메모리가 중요
<img src='https://www.dropbox.com/s/792k9d9m0bgimmq/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.04.31.png?raw=1'>
	* 과거를 잘 기억
	* 이전에 어떤 내용들을 썼는지, 잘 반영할 수 있는 메모리가 필요
	* 이전꺼를 많이 기억하고 있으면 실수가 적게 발생
	* 기본적인 RNN은 동일한 구조로 반복되어, Long-term depends 잡기에 적합하지 않음

* Basic Architecture
<img src='https://www.dropbox.com/s/73359p33r67ot2g/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.04.45.png?raw=1'>
	* 그래서 LSTM을 사용

* Prediction Network
<img src='https://www.dropbox.com/s/vgix7ltym9lg5ej/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.04.54.png?raw=1'>
<img src='https://www.dropbox.com/s/4umuu8tysvvzh9n/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.05.12.png?raw=1'>
	* 어떤 입력이 주어졌을 때, 다음번 입력에 대한 확률 분포를 정의
	* a,b,c가 있는 상황에서 y는 다음번 x에 1,2,3이 있기 때문에 1,2,3의 확률 분포를 정의하면 됨

* Text Prediction
<img src='https://www.dropbox.com/s/jn4mfdo274yeter/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.05.18.png?raw=1'>
	* k개의 글자가 있고, k개 중에 하나를 고름

* Handwriting Prediction
<img src='https://www.dropbox.com/s/detktl2pflwd8tn/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.05.28.png?raw=1'>
<img src='https://www.dropbox.com/s/bdwtpryksqx8iru/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.06.20.png?raw=1'>
	* Mixture Density Network를 써야 함 -> 이유 : countinues하기 때문에, 더이상 디스크립트 한 값들의 연속으로 표현할 수가 없어, 다음번 확률 분포가 가우시안 믹스쳐 분포를 만들어야 함
	* 참고 - [Mixture of Gaussians](http://norman3.github.io/prml/docs/chapter09/2.html) : 여러 개의 가우시안 분포를 선형 결합한 형태
	* 참고 - [머신러닝 - 수식 없이 이해하는 Gaussian Mixture Model : GMM](https://3months.tistory.com/154)
		* 3가지 정규분포 중 확률적으로 어디에 속해 있는가를 나타내는 Weight 값 = 잠재변수
		* 각각의 정규분포의 모수(평균,분산)
	* k개의 weight, mean, variance, (추가포함) end of stroke : 손글씨를 쓰고 멈출지? 안멈출지?
	* weight, mean, variance, correlation 파라미터를 통해 하나의 Gaussian Mixture Modeling
<img src='https://www.dropbox.com/s/jhgwfgd6gn7qpnu/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.06.58.png?raw=1'>
<img src='https://www.dropbox.com/s/4g7chvzennnax9h/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.07.05.png?raw=1'>
	* `x_t`: Input data가 컨티뉴어스하더라도 충분히 이 모델을 이용해서 잘 만들 수가 있음
	* Under 글씨가 잘 예측된 결과

* Handwriting Synthesis
<img src='https://www.dropbox.com/s/l17uobevrvajoz2/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.07.10.png?raw=1'>

## 4. Experimental Results
* Handwriting Synthesis
<img src='https://www.dropbox.com/s/nm9eors45cmlh6z/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.07.54.png?raw=1'>
	* 캐릭터가 주어지면, 캐릭터에 맞는 글씨를 쓰는 네트워크를 학습하기를 원함
	* a,b,c를 쓰기 위한 stroke의 개수는 30개, 50개일 수도 있음 -> 2개가 숫자가 차이나기 때문에 맵핑을 찾아서는 안됨
	* Soft Attention = soft window 구조가 추가되어 사용

* Synthesis Network
<img src='https://www.dropbox.com/s/qbmer8o84yszh6r/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.08.01.png?raw=1'>
<img src='https://www.dropbox.com/s/rtxx3p0qp8idssp/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.08.07.png?raw=1'>
	* 캐릭터들이 주어졌을 때, window를 이용해서 output 생성
	* `input = d_x, d_y, stroke`
	* `output = d_x, d_y, pen displacement를 GMM으로 모델링 하고, end of stroke를 ??? 모델링하는 이상한 확률 분포의 파라미터`
<img src='https://www.dropbox.com/s/t5shfiis5szpyzn/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.08.27.png?raw=1'>
<img src='https://www.dropbox.com/s/sr32row6inpr3im/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.08.36.png?raw=1'>
	* soft window = soft attention과 동일한 구조
	* pi = u개의 각각의 캐릭터가 t번째 시간에 얼마나 중요한지? t번째 시간에 u번째 weight가 얼마나 중요한지에 대한 weight 역할을 수행
	* 카파 = 지금 시간에 몇 번째 단어를 집중하는지 나타냄, 이전 카파에 지금 뉴럴넷에 나온 것을 더함
	* 소프트 윈도우 = a, b, c 단어를 조금씩 전진
	* 알파, 베타 = 중요도와 폭을 결정

* Heuristics
<img src='https://www.dropbox.com/s/z15k1npaml06wvs/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.08.49.png?raw=1'>
<img src='https://www.dropbox.com/s/omz132q7mzopuwl/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.08.59.png?raw=1'>
	* 언제까지 캐릭터를 써야 할까요? -> 주어진 단어가 5개인데, 6번째가 5번째보다 더 많이 나오면(확률이 커지면) 멈춤
	* biased sampling : 가우시안을 샘플링하게 되면, 이상한 것이 나오면 망가지게 됨 -> 확률 분포를 mean 주변으로 나오게 하는 방법 (대신에 정형화된 글씨가 써지게 됨)

* Biased Sampling
<img src='https://www.dropbox.com/s/pwcvgzcipz7f0du/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.09.11.png?raw=1'>
	* biased를 키우면 정형화된 글씨, 낮추면 필기체와 같은 글씨를 써지게 됨


## 5. Conclusion
* 결론
<img src='https://www.dropbox.com/s/ond5gd6jt6jmgfn/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7%202019-01-06%2023.09.32.png?raw=1'>
	* lstm은 되게 중요함
	* RNN은 deterministic model = 입력이 고정되면 출력이 고정
	* 스토케스팅 모델을 만듬
	* soft window, generation model이 섞임으로써 sequences generation 할 수 있게 됨
	* biased로 글씨체를 결정

---

### References

* edwith - [논문으로 짚어보는 딥러닝의 맥 - RNN을 이용해 손글씨를 만드는 Handwriting generation](https://www.edwith.org/deeplearningchoi/lecture/15843/)
* 참고논문 - [Generating Sequences With Recurrent Neural Networks(2014)](https://arxiv.org/pdf/1308.0850.pdf)
* github - [rnnlib](https://github.com/szcom/rnnlib)
