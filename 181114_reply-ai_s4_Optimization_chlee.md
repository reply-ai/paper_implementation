## Optimization 방법론

### 학습 목표
Neural Net을 학습시키는 training을 할 때 사용하는 방법을 Optimization, 최적화 작업이라고 합니다.

이런 Optimization 방법에는 어떤 것들이 있고 어떻게 발전되어 왔는지 알아보도록 하겠습니다.

이 내용의 필요성이 체감적으로 낮을 수 있지만, 실질적으로 굉장히 유용한 기능이기 때문에 

Optimization에 대해 알아두는 것이 도움이 될 것입니다.

---

#### 1-1. Gradient Descent

* 어떤 함수가 있으면 그의 최소점을 찾고자 함
* 함수가 살고 있는 공간은 파라미터
	* 파라미터가 2라면, 2차원 공간 --> 그림을 그려보자!
	* 파라미터가 1억개라면? --> 불가능

--> 이것들의 기울기만 알고 있다면? G_w의 cost가 최소인 function을 찾고자 함
* gradient가 1차 미분
	* 그렇다면 2차 미분은(기울기의 기울기)? 해시안?
	* 뉴턴 메소드? 어떤 위치에서의 기울기를 찾을 때 혹은 이 위치에 최소로 가는 점을 찾을 때, 여기서의 2차 근사를 하여, 2차 근사한 위치로 x를 옮김 --> 계산하기 힘든 이유 : 2차 근사를 위해선 파라미터가 100개면 gradient dimension이 100개 (각 각 편미분하면 구할 수 있음)
	* 해시안 = 100 x 100 

<img src='https://www.dropbox.com/s/xkmx2hbqyzmmonc/1.png?raw=1'>

#### 1-2. Gradient 종류

* Batch gradient descent
	* train data : 55,000개 있다면 한번에 고려해서 한번 gradient 계산하는 것
	* 55,000개를 각각 계산하고 평균내서 한번 업데이트
	* <span style="color:blue">단점 : 오래 걸림</span>

* Mini-Batch gradient descent
	* 보통 64 ~ 256개를 사용, 번갈아가면서 골라서 학습 (2의 n승 단위로 진행 : gpu를 활용할 경우, gpu가 메모리를 할당하는 단위가 2의 n승이기 때문에)
	* <span style="color:blue">batch size가 작으면 작을수록 성능이 높으나, 시간이 오래 걸림</span>

<img src='https://www.dropbox.com/s/rnepmybx73md8hg/3.png?raw=1'>

* Stochastic gradient descent
	* Batch gradient descent와 반대로, 한번에 1개씩 계산
	* 1개씩만 계산하기 때문에, 떨림이 심함
	* random으로 하나를 뽑는데 expectition이 원래 batch gradient descent와 같아서 Stochastic gradient descent로 불림

<img src='https://www.dropbox.com/s/uri525esjh8eeco/2.png?raw=1'>

	
#### 1-3. Gradient Challenges

* learning rate 설정이 어려움
--> 가고가 하는 방향으로 가면 줄어 든다는 것은 알고 있으나, 얼마나 가야 할지를 모름
* 보통 line search라고 하여, 일일이하는 방법도 있음(log scale로 진행) --> learning rate schedule를 진행

<img src='https://www.dropbox.com/s/t7vdg2pzilvgb7d/4.png?raw=1'>

* Local minima에 빠지게 됨

#### 1-4. Momentum

* 현재의 momentum : 이전에 얻어진 gradient를 현재 gradient를 더함 --> 마치 관성이 있는 것처럼 계산 : Momentum

<img src='https://www.dropbox.com/s/h1897oa3nbee60r/5.png?raw=1'>

<img src='https://www.dropbox.com/s/ei0geq40v5npesn/6.png?raw=1'>

* Nesterov accelerated gradient : gradient를 momentum 만큼 이동해서 gradient 계산
<img src='https://www.dropbox.com/s/oqrv5an1x6y2mgl/7.png?raw=1'>
<img src='https://www.dropbox.com/s/ojvfufdng74878q/8.png?raw=1'>
	* 현재의 momentum을 할 경우에는 계속 더하면서 진행하기 때문에 global minimize하지 못하고 계속 왔다갔다하는 경우가 발생할 수 있음
	* <span style="color:blue">Nesterov accelerated는 이동할 graident에서 진행하기 때문에 global minimize를 못하고 왔다갔다하는 상황을 방지 할 수 있음</span>
	* tensorflow - [Momentum Nesterov 링크](https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer#top_of_page)
	* torch - [Momentum Nesterov 링크](https://pytorch.org/docs/stable/optim.html)

#### 2-1. Adagrad

* 모든 파라미터마다 learning rate를 변경
* 파라미터가 10,000개라면 보통 learning rate는 1개지만, learning rate도 10,000개를 변경하여 적용 (많이 변했던 파라미터는 조금 변하게 하고, 조금 변한건 조금 변하게 함
* G : 지금까지 gradient의 곱
* \Eta : learning rate에서 권장하는 파라미터, 원래 \Eta만 존재

<img src='https://www.dropbox.com/s/gb6z67ypk2hmwj1/9.png?raw=1'>
<img src='https://www.dropbox.com/s/srxj42l8z7tdmnj/10.png?raw=1'>

#### 2-2. Adadelta

<img src='https://www.dropbox.com/s/5guoo8tyxdwh03v/11.png?raw=1'>

* 단점 : G가 계속 커지는 경우 --> learning rate는 계속 줄어들어 해당 네트워크는 학습이 저장되지 않음
* 해결방법 : exponential moving average을 취함
	* \gamma와 1-\gamma를 가지고 어떤 값을 업데이트 함
	* g_t^2 (지금 들어온 gradient의 제곱)
	* g_t^2이 한동안 안커지면 g^2이 작아지게 됨, 마치 윈도우 사이즈만큼에 g^2을 더한 효과가 생김
		* E[g^2]_t : gradient = 분모
		* E[\Delta\theta^2]_t : graident가 변한 값 = 분자
	* \varepsilon : 초기에 분자의 값이 커지는 것을 막기 위해 존재 (E[g^2]_t가 초기에는 0이기 때문 / 0으로 나누면 무한대가 되기 때문에)

#### 2-3. RMSprop

* Adadelta와 매우 비슷하나, learning rate가 존재한다는 점이 차이점
* 힐튼 교수의 코세라 강의에 칠판에 적혀진걸 사람들이 사용해서 퍼지게 됨

##### 차이점 정리
* <span style="color:blue">**Adagrad는 exponential moving average를 쓰지않고, gradient 변화량의 제곱을 다 더했기 때문에, adaptive learning 가능하지만, learning는 계속 줄어 들게됨**</span>

* <span style="color:blue">**Adadelta exponential moving average를 사용하고 최근 윈도우가 graident가 변했는지 체크**</span>
* <span style="color:blue">**RMSprop는 global learning rate가 있다는게 차이점**
</span>

#### 2-4. Adam

<img src='https://www.dropbox.com/s/n19g5em6f7p6lkk/12.png?raw=1'>
<img src='https://www.dropbox.com/s/jo5a9ickpkf4usk/13.png?raw=1'>

* learning와 Momentum을 합친 것
* m_t : momentum 변화량
* **b_1(중요)** : momentum을 구할 때 얼마나 exponetial moving average 할지?
* v_t : graident 변화량
* b_2 : adaptive learning를 위해서,  얼마나 graident^2을 exponetial moving average 할지? --> 보통 0.9 이상 사용
* **\varepsilon(중요)** 
* 보정하는 term(bais-correction) : \root 1-\beta_2^t / 1-\beta_1^t

<img src='https://www.dropbox.com/s/svr5c5hjavgzzcd/14.png?raw=1'>

* 초기 파라미터 : epsilon = 1e-08인데 적절하지 않다고 생각
* epsilon을 조금 키워하는게 효율적 : 1e-04 or 1e-02 한게 효과적인 것으로 경험, regression 문제에서 더 키운게 효과적이였음
**왠만한 momentum에서는 Adam 추천**

#### Visualization

<img src='http://cs231n.github.io/assets/nn3/opt2.gif?raw=1'>

<img src='http://cs231n.github.io/assets/nn3/opt1.gif?raw=1'>

* local minimal의 경우 SGD는 잘 못 벗어나는 반면, adaptive 방법론은 빨리 벗어나게 됨 --> 벗어나는 방향으로 조금의 정보가 들어오면 지금까지 안 움직이기 때문에 확 밀어버려서 빨리 벗어나게 됨

#### Result

<img src='https://www.dropbox.com/s/49mndodmzxq8x81/15.png?raw=1'>

* **adaptive 강추**
* adagrade learning가 줄어드는 효과가 있어, RMSprop를 발전 시킴 = Adadelta
* bais-correction = Adam 보정하는 term
* Shuffling : 학습 data는 매 epoch마다 섞는게 효과적

<img src='https://www.dropbox.com/s/bgk0rics2svn6gu/16.png?raw=1'>

* Curriculum Learning : 조금 쉬운 데이터를 가지고 먼저 학습 시키고, 나중에 어려운 데이터를 학습

<img src='https://www.dropbox.com/s/v8x51od2q67lkce/17.png?raw=1'>

* **batch normalization(매우 중요)**
 * 논문 - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf)
 * Batch Normalization은 기본적으로 Gradient Vanishing / Gradient Exploding 이 일어나지 않도록 하는 아이디어 중의 하나
 * Algorithm
 	- mini-batch 단위 학습할때, mini-batch 단위 feature 별로 평균과 표준편차를 구해준 다음 normalize (x\hat) 해주고, scale factor와 shift factor를 이용하여 새로운 값을 만듬
 	- \beta, \gamma 
<img src='https://shuuki4.files.wordpress.com/2016/01/bn1.png?raw=1'>
	- Hidden Layer에 들어가기 전에 Batch Normalization Layer를 더해주어 input을 modify 해준 뒤 새로운 값을 activation function을 넣어주는 방식 사용
<img src='https://i0.wp.com/mohammadpz.github.io/posts/2015_02_01_IFT6266_Cats_vs_Dogs/img/bn.png?zoom=2?raw=1'>
	- Train Data는 mini-batch에서 평균과 표준편차를 구해 normalization 하고, Test Data는 training 할 때 이동평균(moving average) 및 unbiased variance estimate의 이동평균을 저장한 값을 이용해서 normalization 함
	- Scale, Shift factor는 다른 레이어에서 weight를 학습 하듯이 back-prop에서 학습된다는 흐름
<img src='https://shuuki4.files.wordpress.com/2016/01/bn2.png?raw=1'>
  
 * 장점

> 기존 Deep Network에서는 learning rate를 너무 높게 잡을 경우 gradient가 explode/vanish 하거나, 나쁜 local minima에 빠지는 문제가 있었다. 이는 parameter들의 scale 때문인데, Batch Normalization을 사용할 경우 propagation 할 때 parameter의 scale에 영향을 받지 않게 된다. 따라서, learning rate를 크게 잡을 수 있게 되고 이는 빠른 학습을 가능케 한다.

> Batch Normalization의 경우 자체적인 regularization 효과가 있다. 이는 기존에 사용하던 weight regularization term 등을 제외할 수 있게 하며, 나아가 Dropout을 제외할 수 있게 한다 (Dropout의 효과와 Batch Normalization의 효과가 같기 때문.) . Dropout의 경우 효과는 좋지만 학습 속도가 다소 느려진다는 단점이 있는데, 이를 제거함으로서 학습 속도도 향상된다. 

<img src='https://shuuki4.files.wordpress.com/2016/01/bn3.png?raw=1'>


* Early stopping

<img src='https://www.dropbox.com/s/aakuc1hp1jod1ux/18.png?raw=1'>

* Gradient noise : gradient에 noise를 섞어주면 학습이 빨리 됨

* Learning rate
	* low learning rate : 멈추고 learning rate를 키워야 함
	* very high learning rate : learning rate를 줄여야 함 / Adam 파라미터를 쓰고 있다면, \varepsilon 키우는 것도 추천

<img src='https://www.dropbox.com/s/4oawuhnnfh6feot/19.png?raw=1'>


---

### References
* edwith - [논문으로 짚어보는 딥러닝의 맥](https://www.edwith.org/deeplearningchoi/lecture/15303/)
* CS231n - [CNN for Visual Recognition](http://cs231n.github.io/neural-networks-3/)
* BEOMSU KIM Blog - [Batch Normalization 설명 및 구현](https://shuuki4.wordpress.com/2016/01/13/batch-normalization-%EC%84%A4%EB%AA%85-%EB%B0%8F-%EA%B5%AC%ED%98%84/)






















