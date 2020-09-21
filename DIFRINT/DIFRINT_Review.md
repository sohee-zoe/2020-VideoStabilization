
# [Deep Iterative Frame Interpolation for Full-frame Video Stabilization](https://deepai.org/publication/deep-iterative-frame-interpolation-for-full-frame-video-stabilization)
JINSOO CHOI and IN SO KWEON, KAIST, Republic of Korea  
[TOG20](https://arxiv.org/pdf/1909.02641.pdf)  /  [ICCVW19](https://ieeexplore.ieee.org/abstract/document/9022415)  /  [github](https://github.com/jinsc37/DIFRINT)  
![](https://www.groundai.com/project/deep-iterative-frame-interpolation-for-full-frame-video-stabilization)

[Video Frame Interpolation -- Interpolation Operations](http://www.visionbib.com/bibliography/motion-i774vin1.html)  



## 1 Introduction

**Existing Approaches**: 
- 3D (Liu et al., [2009](http://gvv.mpi-inf.mpg.de/teaching/gvv_seminar_2012/papers/Content-Preserving%20Warps%20for%203D%20Video%20Stabilization.pdf), [2012](https://ieeexplore.ieee.org/document/6247662); Zhou et al., [2013](https://ieeexplore.ieee.org/document/6619142))
- 2.5D (Liu et al., [2011](http://web.cecs.pdx.edu/~fliu/project/subspace_stabilization/), [2013a](http://web.cecs.pdx.edu/~fliu/project/joint-subspace/); Goldstein and Fattal, [2012](https://www.cse.huji.ac.il/~raananf/projects/stab/paper.pdf))
- 2D (Liu et al., [SteadyFlow 2014](https://ieeexplore.ieee.org/document/6909932?arnumber=6909932), [Meshflow 2016](http://www.liushuaicheng.org/eccv2016/meshflow.pdf)/[poster](http://www.eccv2016.org/files/posters/P-4A-28.pdf)/[github](https://github.com/sudheerachary/Mesh-Flow-Video-Stabilization), [2013b](http://www.liushuaicheng.org/SIGGRAPH2013/index.htm)/[github](https://github.com/SuTanTank/BundledCameraPathVideoStabilization))
- Deep learning-based (Wang et al., [2018](https://github.com/cxjyxxme/deep-online-video-stabilization-deploy); Xu et al., [2018](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13566)/[github](https://github.com/mindazhao/Deep-Video-Stabilization-Using-Adversarial-Networks)) approaches

대부분의 기존 video stabilization methods는 후처리로 비디오를 오프라인에서 안정화시킨다. 오프라인 방법은 일반적으로 온라인 방법에 비해 안정화 결과가 좋지만 최근 딥러닝 기반 접근법은 유망한 품질을 보여주었다. 딥러닝 기반 방법은 supervised approach를 취하므로 unstable (shaky)과 stable (motion smoothed) 비디오 쌍 데이터가 필요하다. 따라서 불안정하고 안정적인 카메라가 동일한 장면을 동시에 캡처한 데이터셋 (Wang et al., [2018](https://ieeexplore.ieee.org/document/8554287))을 활용한다. 또한 딥러닝 방법을 포함한 대부분의 video stabilization approaches는 일시적으로 missing view 때문에 프레임 경계를 crop할 필요가 있다. 카메라 흔들림은 인접한 프레임과 대조적으로 프레임 경계에서 일시적으로 content가 누락되는 원인이 된다 ( (c)). 따라서 대부분의 최신 방법들은 프레임 경계를 자른다. cropping은 content의 손실과 불가피한 줌인 효과를 야기하는데, 비디오 안정화 연구의 주요 목표 중 하나가 과도한 cropping을 줄이는 것이다.

![Fig1](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig1.png)


unsupervised 방식으로 훈련될 수 있어 대응되는 안정적인 GT를 필요로 하지 않는 deep framework를 제안한다. frame interpolation을 사용하여 프레임을 안정화함으로써 사실상 cropping을 제거한다. 기본적으로 deep framework는 두 개의 순차적인 프레임 (sequential frames)의 "중간" frame을 생성하는 것을 배운다. interpolation 관점에서 볼 때, synthesized midde (i.e. interpolated) frame은 두 개의 순차적인 프레임 사이에 캡처되었을 프레임을 나타낸다. 즉, interpolated frame은 프레임 간 모션(inter-frame motion)의 정확한 중간에 걸리는 것으로 가정되는 중간 프레임을 말한다 (Niklaus et al., 2017b; Niklaus and Liu, 2018). 따라서 Fig. 1 (b)와 같이 중간 프레임의 순차적 생성은 인접 프레임 사이의 spatial jitter를 감소시킨다. 직관적으로 frame interpolation은 공간 데이터 시퀀스(spatial data sequences)에 대한 시간 영역에서 liner interpolation (low-pass filter)으로 간주할 수 있다. liner interpolation을 여러 번 사용하면 안정화 효과가 더 강해진다. spatial data sequences (i.e frame sequences)의 경우 interpolation은 기본적으로 모든 픽셀의 정확한 절반 지점을 추정하고 Fig. 2와 같이 중간 프레임을 생성한다. 또한 중간 프레임 합성 (middle frame synthesis)의 주요 장점은 프레임 경계가 프레임 간 카메라 모션 사이에서 합성되어 일시적으로 missing view를 채우고 full-frame 기능을 제공한다. 게다가, deep architecture로 빠른 feed forwarding을 통해 거의 실시간 (15 fps)로 실행할 수 있다.

![Fig2](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig2.png)



**Contributions**:  
1. 거의 실시간으로 cropping 없이 full-frame video stabilization 가능하다.
2. unsupervised manner로 학습하여 안정적인 GT 비디오가 필요하지 않다.
3. frame interpolation 기법을 사용하여 시각적 왜곡 (wobbling artifacts)을 줄인다.


## 2 Related Work

**3D methods**: 안정화를 위해 3D 공간에서 카메라 궤적(trajectory)을 모델링한다.   
- Structure from Motion (SfM) (Liu et al., 2009),   
- depth information (Liu et al., 2012),   
- 3D plane constraints (Zhou et al., 2013),   
- projective 3D reconstruction (Buehler et al., 2001),   
- light field (Smith et al., 2009),   
- 3D rotation estimation via gyroscope (Karpenko et al., 2011; Bell et al., 2014; Ovrén and Forssén, 2015) 


**2.5D approaches**: 3D 모델의 부분 정보를 사용하며, 재구성 실패 차례를 처리할 수 있다.
- Liu et al., 2011: 3D constraints을 유지하기 위해 subspace constraints를 통해 long feature tracks를 smoothing 하는 반면,
- Liu et al., 2013a (extension): stereoscopic videos에 접근 방식을 적용
- Goldstein et al., 2012: epipolar transfer을 통해 feature track 길이를 향상시킨다.

3D 및 2.5D 방법의 장점은 강력한 안정화 효과를 제공한다. 


**2D methods**: 일반적으로 비디오 프레임에 2D linear transform을 적용하여 효율적이고 강건하게 수행된다.
- Early works  
  - Matsushita et al., 2006; Chen et al., 2008: motion smoothing을 위해 lowpass filter를 적용하는 동시에 motion inpainting을 통해 full-frame capabilities를 제공한다.
- Recent works 
  - Grundmann et al., [2011](https://ieeexplore.ieee.org/document/5995525) **L1-Stabilizer**: L1-norm optimization을 적용하여 cinematography motion으로 구성된 smooth camera path를 계산한다.
  - Grundmann et al., 2012: rolling shutter effects를 보상하기 위해 아이디어 확장
  - Liu et al., 2013b: 공간적으로 다양한 모션을 위해 multiple local camera paths의 camera trajectory를 모델링
  - Liu et al., [2014](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.650.7509) **SteadyFlow**: feature tracks 보다 pixel profiles를 스무딩하는 것을 제안
  - Liu et al., [2016](https://link.springer.com/chapter/10.1007%2F978-3-319-46466-4_48) **MeshFlow**: mesh profiles의 보다 효과적인 smoothing으로 확장
  - Liu et al., 2017: video coding으로 확장
  - Huang et al., 2018: camera motion search의 효율성을 개선하기 위해 video coding 사용


**Other techniques**
- Wang et al., 2013: generate smooth camera motion을 위해 spatiotemporal optimization 적용
- Baie et al., 2014: generate smooth camera motion을 위해 user interaction 적용

**Online video stabilization methods**
- Liu et al., 2017, 2016; Wang et al., 2018; Xu et al., 2018: 모델에서 생성된 과거 프레임을 사용하여 현재의 안정화된 파라미터를 추정한다.

  - **Deep learning-based approaches**: 오프라인 방식이 더 나은 결과를 보여주지만 딥러닝 기반 접근 방식은 유망한 결과를 보여준다.
    - 명시적으로 camera path를 계산하는 대신 supervised learning approach을 모델링한다.
    - Wang et al., 2018: video stabilization를 위해 stability와 temporal loss terms를 정의하여 supervised learning framework을 제안한다. 이 작업은 labeled dataset (unstable & stable video sets)이 제공한다.
    - Xu et al., 2018: supervised manner로 학습된 adversarial network을 제안한다. 이 네트워크는 안정화된 프레임을 생성하기 위해 adversarial network를 통해 warping parameters를 추정한다.


DIFRINT는 2D 방법의 장점을 이어받아 deep frame interpolation라는 맥락에서 video stabilization을 수행한다. 이로 인해 왜곡이 최소화되는 반면, copping이 필요로하지 않다. 게다가, unspervised manner로 학습되고, GT 비디오를 필요로 하지 않는다.


## 3 Proposed Method: DIFRINT
이 접근법의 핵심 아이디어는 주어진 비디오의 두 프레임 사이에 순차적으로 중간 프레임을 통합하여 시각적 안정성을 가져오는 것이다. 2D 방법으로 간주될 수 있는데, 각 연속 프레임은 인접 프레임을 활용하여 frame interpolation을 거친다. 본문에서 interpolated frame과 middle frame은 동의어로 사용된다.  


이 방법은 여러 번 반복해서 interpolation을 적용하면 안정화 효과를 높일 수 있다. 그러한 방법을 구현하는 가장 간단한 방법은 단순히 deep frame interpolation architecture를 고안하는 것이다. Fig. 3 (b)에서 볼 수 있듯이, 두 프레임이 주어지면, bidirectional optical flow을 통해 서로 _반절씩_ (×0.5로 표시됨) warping 된다 (PWC-Net [Sun et al. 2018]).  
> they are warped 'toward' each other _halfway_ (denoted as ‘×0.5’) via bidirectional optical flow (PWC-NET).  

좀 더 구체적으로 말하면, ![equ](https://latex.codecogs.com/gif.latex?f_{i-1})은 ![equ](https://latex.codecogs.com/gif.latex?f_{i+1})의 반절 (halfway)로 warping 되는 반면, ![equ](https://latex.codecogs.com/gif.latex?f_{i+1}) 또한 ![equ](https://latex.codecogs.com/gif.latex?f_{i-1})의 반절 (halfway)로 warping 되어서 warped frames ![equ](http://latex.codecogs.com/svg.latex?f_{w}^{-})와 ![equ](http://latex.codecogs.com/svg.latex?f_{w}^{+})를 생성한다. 즉, ![equ](http://latex.codecogs.com/svg.latex?f_{w}^{-})와 ![equ](http://latex.codecogs.com/svg.latex?f_{w}^{+})는 ![equ](https://latex.codecogs.com/gif.latex?f_{i-1})와 ![equ](https://latex.codecogs.com/gif.latex?f_{i+1})에서 각각 발생하는 중간 지점을 나타낸다. 그런 다음, 중간 지점 프레임들 (halfway point frames) ![equ](http://latex.codecogs.com/svg.latex?f_{w}^{-})와 ![equ](http://latex.codecogs.com/svg.latex?f_{w}^{+})를 CNN (U-Net architecture)에 입력으로 하여 중간 프레임 (middle frame) ![equ](http://latex.codecogs.com/svg.latex?f_{int})를 생성한다. U-Net module은 다양한 스케일의 정보가 어떻게 결합되어야 하는지를 배울 수 있으며, global low-resolution information으로 local high-resolution을 예측할 수 있게 한다. warped images는 holes or unseen regions을 포함할 수 있으므로 서로 보완을 위해 ![equ](http://latex.codecogs.com/svg.latex?f_{w}^{-})와 ![equ](http://latex.codecogs.com/svg.latex?f_{w}^{+}) 둘 다 사용하여 CNN을 학습한다.  


위에서 설명한 deep frame interpolationd 방법은 Fig. 4와 같이 여러 번 반복하면 blur가 누적된다. 이는 여러 번 반복을 통해 미세한 디테일에 대한 정보가 손실되기 때문이다. blur accumulation을 방지하기 위해 ![equ](http://latex.codecogs.com/svg.latex?f_{int}) 쪽으로 warping된 original frame ![equ](http://latex.codecogs.com/svg.latex?f_{i})와 ![equ](http://latex.codecogs.com/svg.latex?f_{int})를 함께 다른 CNN (ResNet architecture)의 입력으로 하여 final middle frame ![equ](http://latex.codecogs.com/svg.latex?\hat{f}_{i})을 생성한다. original frame ![equ](http://latex.codecogs.com/svg.latex?f_{i})는 모든 iteration에 사용된다. 따라서 여러 번 반복해도 original frame에 미세한 detail이 포함되어 있기 때문에 resampling error가 발생하지 않는다. ResNet 구조는 residual learning mechanism을 통해 세부적인 디테일에서 에러를 최소화하는 데 매우 적합하다.  
요약하자면, frame interpolation은 CNN에 제공된 두 프레임을 서로 반쯤 warping 하고 CNN에 입력으로 넣어 ‘intermediate frame’ ![equ](http://latex.codecogs.com/svg.latex?f_{int})를 생성한다. 그리고 나서, ![equ](http://latex.codecogs.com/svg.latex?f_{int})는 original frame ![equ](http://latex.codecogs.com/svg.latex?f_{i})이 warping 된 것을 참고하여 다른 CNN에 ![equ](http://latex.codecogs.com/svg.latex?f_{int})와 함께 입력으로 하여 최종적으로 interpolated frame ![equ](http://latex.codecogs.com/svg.latex?\hat{f}_{i})을 생성한다.

![Fig3](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig3.png)
![Fig4](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig4.png)


### 3.1 Unsupervised learning framework
이 접근방식의 주요한 장점은 unsupervised learning을 통한 video stabilization을 해결하기 위해 deep learning과 frame interpolation을 이용한다는 것이다.

#### 3.1.1 Training scheme
제안된 프레임워크의 목적은 error accumulation 없이 중간 프레임을 생성하는 것이다. 따라서 training scheme의 목표는 그러한 interpolation quality를 산출할 수 있도록 프레임워크를 적절하게 training 하는 것이다. training scheme을 이해하려면 먼저 test scheme을 이해해야 한다. 위에서 설명한 것처럼, 실제 interpolation은 두 프레임을 서로 반쯤 warping 하고 U-Net 구조를 통해 입력으로 넣어 ![equ](http://latex.codecogs.com/svg.latex?f_{int})를 생성함으로써 구현된다. error나 blurring을 방지하기 위해 original frame ![equ](http://latex.codecogs.com/svg.latex?f_{i})을 ![equ](http://latex.codecogs.com/svg.latex?f_{int}) 쪽으로 warping 해서 ResNet 구조를 통해 입력으로 넣어 ![equ](http://latex.codecogs.com/svg.latex?\hat{f}_{i})을 생성한다. 이 모델을 학습할 때 문제가 발생한다. 즉, GT middle frame이 존재하지 않는다. 인접한 프레임 ![equ](http://latex.codecogs.com/svg.latex?f_{i-1})와 ![equ](http://latex.codecogs.com/svg.latex?f_{i+1}) 사이의 중간 지점이 되는 것을 보장하지 않기 때문에 단순히 ![equ](http://latex.codecogs.com/svg.latex?f_{i})를 GT로 사용할 수 없다. 따라서, ![equ](http://latex.codecogs.com/svg.latex?\hat{f}_{i})은 loss를 비교하고 계산할 수 있는 근거가 없다.  


이 시점에서, original frame ![equ](http://latex.codecogs.com/svg.latex?f_{i})의 공간적으로 translated version인 pseudo-GT frame ![equ](http://latex.codecogs.com/svg.latex?f_{s})를 정의한다. Translation은 small random scale (frame width의 최대 1/8)로 임의 방향으로 수행된다. 학습은 인접한 프레임의 ![equ](http://latex.codecogs.com/svg.latex?f_{i-1})과 ![equ](http://latex.codecogs.com/svg.latex?f_{i+1})을 ![equ](http://latex.codecogs.com/svg.latex?f_{s}) 방향으로 warping하고 ![equ](http://latex.codecogs.com/svg.latex?f_{s})를 _reconstruct_ 하는 것을 목표로 수행된다. 이러한 방법으로 U-Net은 두 개의 warped frames ![equ](http://latex.codecogs.com/svg.latex?f_{w}^{-})와 ![equ](http://latex.codecogs.com/svg.latex?f_{w}^{+})으로 ![equ](http://latex.codecogs.com/svg.latex?f_{s})를 reconstruct 하는 방법을 배우며, ResNet은 warped original frame ![equ](http://latex.codecogs.com/svg.latex?f_{i})으로 동일한 작업을 수행한다. 이러한 training scheme은 testing scheme에 따라 일반화 (generalize) 되며, _virtual middle frame_을 reconstructing 하는 것으로 볼 수 있다. training scheme의 개요는 Fig. 3 (a)에 나타나 있다.




#### 3.1.2 Testing scheme  
일단 training scheme이 pseudo-GT ![equ](http://latex.codecogs.com/svg.latex?f_{s})를 reconstructing 함으로써 모델을 적절하게 학습시키면, actual frame interpolation을 test scheme에 적용할 수 있다. 인접 프레임 ![equ](http://latex.codecogs.com/svg.latex?f_{i-1})와 ![equ](http://latex.codecogs.com/svg.latex?f_{i+1})은 Fig. 3 (b)와 같이 서로를 향해 0.5 비율로 반쯤 warping 된다. frame interpolation 작업에는 연속 프레임을 반절씩 warping 하거나 중간 프레임을 에측하는 방법을 배우는 기술이 사용되었다 ([Meyer et al. 2015; Niklaus and Liu 2018; Niklaus et al. 2017a,b). 실제로 비디오 시퀀스를가 주어지면 연속 프레임 3개 (![equ](http://latex.codecogs.com/svg.latex?f_{i-1}), ![equ](http://latex.codecogs.com/svg.latex?f_{i}), ![equ](http://latex.codecogs.com/svg.latex?f_{i+1}))를 프레임워크 입력으로 사용하여 stabilized frame outputs을 생성하고 첫 번째 프레임과 마지막 프레임은 그대로 유지한다. 이미 interpolated frames에 대해 반복적으로 방법을 적용하여 생성된 비디오 프레임을 더욱 stabilize 하는 옵션을 제공한다.  


Iterative frame interpolation으로 시각적 안정성이 강화된다. Fig. 5에서 볼 수 있듯이, ![equ](http://latex.codecogs.com/svg.latex?\hat{f}_{i}^{2})는 두 번의 반복을 거쳤기 때문에, 공간 방향 (spatial orientation)은 ![equ](http://latex.codecogs.com/svg.latex?f_{i-2})에서 ![equ](http://latex.codecogs.com/svg.latex?f_{i+2})까지 영향을 받는 반면, ![equ](http://latex.codecogs.com/svg.latex?\hat{f}_{i}^{1})을 생성하는 한 번 반복은 ![equ](http://latex.codecogs.com/svg.latex?f_{i-1})와 ![equ](http://latex.codecogs.com/svg.latex?f_{i+1})에만 영향을 받는다. 따라서, Fig. 6과 같이 반복 횟수가 많을수록 더욱 global stabilization 된다. 여기서, 안정성을 조정하기 위한 또 다른 파라미터, 즉 interpolation에 사용할 프레임을 수정하는 skip parameter를 제공한다. 예를 들어, default interpolation은 ![equ](http://latex.codecogs.com/svg.latex?f_{i-1})와 ![equ](http://latex.codecogs.com/svg.latex?f_{i+1})을 인접 프레임 (skip = 1)으로 설정하년 반면, skip parameter를 2로 설정하면 ![equ](http://latex.codecogs.com/svg.latex?f_{i-2})와 ![equ](http://latex.codecogs.com/svg.latex?f_{i+2})를 입력으로 사용한다. 인접 프레임 (e.g., 첫 번째 또는 마지막 프레임에 가까움)을 건너 뛰지 않은 세 프레임의 경우, 더 작은 skip parameter가 할당된다.

![Fig5](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig5.png)
![Fig6](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig6.png)


#### 3.1.3 Loss functions  
네트워크 구성 요소를 학습하기 위해 pixel-wise color-based loss function을 사용한다. ![equ](http://latex.codecogs.com/svg.latex?l^{2})-loss는 흐릿한 (blurry) 결과를 만드는 것으로 보고되었기 때문에, 다음과 같이 정의된 ![equ](http://latex.codecogs.com/svg.latex?l^{1})-loss function을 사용한다.   

(1) ![equ](https://latex.codecogs.com/gif.latex?L_1%3D%5Cleft%20%5C%7C%20f_s-%5Chat%7Bh%7D_i%20%5Cright%20%5C%7C_1)  


여기서 ![equ](http://latex.codecogs.com/svg.latex?L_{1})은 pseudo-GT frame ![equ](http://latex.codecogs.com/svg.latex?f_{s})와 output frame ![equ](http://latex.codecogs.com/svg.latex?\hat{f}_{i}) 사이의 loss이다. 
또한 VGG-19의 relu4_3 layer의 반응을 이용하여 perceptual loss function을 고려한다.  

(2) ![equ](https://latex.codecogs.com/gif.latex?L_p%20%3D%20%5Cleft%20%5C%7C%20%5Cphi%20%5Cleft%20%28%20f_%7Bs%7D%20%5Cright%20%29%20-%20%5Cphi%20%5Cleft%20%28%20%5Chat%7Bf%7D_%7Bs%7D%20%5Cright%20%29%20%5Cright%20%5C%7C_2%5E2)  

여기서 ![equ](https://latex.codecogs.com/gif.latex?%5Cphi)는 feature vector를 나타낸다.
final loss는 ![equ](http://latex.codecogs.com/svg.latex?L_{1})과 ![equ](http://latex.codecogs.com/svg.latex?L_{p})의 합이다.   

(3) ![](https://latex.codecogs.com/gif.latex?L_%7Bout%7D%3DL_1%20&plus;%20L_p)  

final loss ![equ](http://latex.codecogs.com/svg.latex?L_{out})은 전체 네트워크를 학습하기에 충분하지만, 동일한 loss를 ![equ](http://latex.codecogs.com/svg.latex?f_{int})에 적용하여 학습 속도를 높이고 성능을 향상시킬 수 있다는 사실을 알아내었다.  

(4) ![](https://latex.codecogs.com/gif.latex?L_%7Bint%7D%20%3D%20%5Cleft%20%5C%7C%20f_s%20&plus;%20f_%7Bint%7D%20%5Cright%20%5C%7C_1%20&plus;%20%5Cleft%20%5C%7C%20%5Cphi%20%5Cleft%20%28%20f_s%20%5Cright%20%29%20-%20%5Cphi%20%5Cleft%20%28%20f_%7Bint%7D%20%5Cright%20%29%20%5Cright%20%5C%7C_2%5E2)  

![equ](http://latex.codecogs.com/svg.latex?f_{int})는 본질적으로 ![equ](http://latex.codecogs.com/svg.latex?f_{s})를 reconstruct 하는 것을 목표로 하기 때문에 이 loss를 적용하는 것이 안전하며, 실제로 U-Net 구성 요소를 명시적으로 학습한다.

![Fig7](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig7.png)
![Fig8](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig8.png)




### 3.2 Implementation
* framework: PyTorch
* 2개의 NVIDIA Titan X (Maxwell)로 하루 학습
* 1280x720 프레임은 0.07초가 걸려 거의 실시간으로 15fps를 생성  
generation process에는 3개의 optical flow estimations, 3개의 warping layers, 그리고 U-Net과 ResNet 구조를 통한 feed forwarding이 모두 포함된다. CUDA의 multi-stream 기능을 사용하면 이전 iteration에서 입력할 두 프레임을 생성한 경우 병렬로 프레임을 생성할 수 있다.  


제안된 프레임워크 내에서 optical flow estimator (PWC-Net)은 고정되어있고 (학습 가능하지 않음), 반면에 학습 가능한 네트워크 U-Net과 ResNet은 고속으로 탁월한 품질을 생산하도록 설계되어 있다. Fig. 7에서 볼 수 있듯이, U-Net 아키텍처는 scaled features 중 3개의 skip connections (점선 화살표)을 사용한다. 아키텍처에는 3×3 convolutional layers가 있는 반면, hidden feature channels는 size가 32, 2의 factor로 down/upscaling 된다. 
U-Net은 intermediate frame ![equ](http://latex.codecogs.com/svg.latex?f_{int})를 생성하고, original frame ![equ](http://latex.codecogs.com/svg.latex?f_{i})가 ![equ](http://latex.codecogs.com/svg.latex?f_{int})쪽으로 warping 되어 ![equ](https://latex.codecogs.com/gif.latex?%5Ctilde%7Bf%7D_i) (초록색)으로 표시된다. 이것들은 concatenate 되고 5개의 residual blocks (파란색)을 가진 ResNet의 입력이 된다. residual blocks은 1×1 convolutional layers (channel size 32)를 사용하여 final output ![](https://latex.codecogs.com/gif.latex?%5Chat%7Bf%7D_i)을 reconstruction 하는 동안 인접 픽셀의 노이즈를 최소화 한다. 또한 1×1 kernels은 feed forward process 속도를 높인다. 게다가, 학습 가능한 전체 구성요소의 all convolutional layers는 gated convolution [Yu et al., 2018] (3×3)을 사용한다. gated convolution은 inpainting task에서 우수한 품질을 보여준다. gated convolution는 각 픽셀 위치와 각 채널에서 동적으로 feature를 선택할 수 있다. gated convolution는 warping 된 후, 구멍이나 unseen regions이 있는 입력 이미지 문제에 매우 적합하다.  


#### 3.2.1 Training settings
learning rate가 0.001이고, mini-batch size가 16 샘플, ![](https://latex.codecogs.com/gif.latex?%5Cbeta%20_1%3D0.9), ![](https://latex.codecogs.com/gif.latex?%5Cbeta%20_2%3D0.999)인 Adam optimizer를 사용하여 256×256 크기의 프레임에 대해 전체 프레임워크를 학습한다. DAVIS dataset [Perazzi et al. 2016]를 사용한다. 이 데이터셋은 다양하고 역동적인 장면들을 담고있고, DIFRINT는 어떠한 GT video set도 필요하지 않기 때문이다. 약 8,000개의 training examples를 활용하여 아키텍처를 200 epoch동안 학습하며, 100 epoch부터 linear decay가 적용된다. 잠재적인 데이터셋 bias를 없애기 위해 full original frames에서 256 × 256 크기의 랜덤 패치를 선택하여 학습 데이터를 augment 한다. 또한 각 패치에 수평 and/or 수직 flip을 랜덤으로 적용한다.

#### 3.2.2 Testing settings
테스트 하는 동안 전체 해상도 프레임이 프레임워크에 입력되어 안정된 프레임 시퀀스를 생성한다. 이 프레임워크가 더 큰 full resolution image를 더 잘 일반화하는 것으로 관찰되었다. 기본적으로 모든 결과에서 iteration 횟수를 5로 정하고 skip parameter는 2로 설정하였다.

## 4 Experiments
방법을 철저히 evaluate 하기 위해, SOTA 방법과 광범위한 비교를 수행한다. 시각적 비교, ablation study 시각화를 제공한다. 마지막으로 상업적 방법에 대한 사용자 선호도 연구를 수행한다. 640 × 360, 1280 × 720, 그리고 1920 × 1080 해상도의 결과가 포함된 supplementary video에서 full video results를 찾을 수 있다.

![Fig9](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig9.png)


### 4.1 Quantitative evaluation
Video stabilization methods는 일반적으로 세가지 지표: _cropping_ _ratio,_ _distortion value,_ _stability score_로 평가된다. 각 메트릭에 대해 좋은 결과는 1.0에 가깝다.  

**Cropping ratio**은 missing view (black) boundaries를 잘라낸 후, 나머지 이미지 영역을 측정한다. 비율이 크면 cropping 횟수가 줄어들기 때문에 비디오 품질이 향상된다. 각 프레임의 입력과 출력 사이의 호모그래피가 맞춰지고 스케일 구성요소가 추출된다. 전체 비디오의 average scale은 cropping ratio metric을 산출한다.  

**Distortion value**은 입력 프레임과 출력 프레임 사이의 호모그래피의 anisotropic scaling을 제공한다. 이 값은 호모그래피의 affine 부분에서 가장 큰 두 고유값의 비율로 계산할 수 있다. 모든 프레임에 대한 호모그래피들 중에서 worst ratio가 distortion value metric으로 선택된다.  

**Stability score**는 출력 비디오의 전반적인 부드러움을 측정한다. 이 메트릭의 경우 주파수 도메인 분석이 사용된다 [Liu et al. 2013b]. camera path는 안정성 값을 계산하는 데 사용된다. 변환 및 회정 구성 요소를 추출하여 두 개의 1D profile signals를 만들 수 있다. 최저 (2~6번째) 주파수 에너지의 합과 총 에너지의 비유을 계산하고 최소값을 취하여 최종 stability score를 얻는다.    



#### 4.1.1 Quantitative results

우리는 첨단 기술과 상업용 알고리즘을 포함한 총 13개의 기준선을 비교했다. 우리는 총 25개의 공개적으로 이용 가능한 비디오 [Liu et al. 2013b]를 평가에 사용했다. 공간 제한으로 인해 그림. 8의 결과와 비교하여 상위 성과 baseline(including one deep online method and three 2D methods)의 대표적인 12개 결과를 보여 준다. 대표 결과는 상위 성과 기준선과의 비교를 가장 많이 포함하고 있기 때문에[Liu et al. 2017]다음과 같이 선정되었다. 그에 따라 결과를 제시한다. 25개의 비디오를 사용하여 13개의 기준선과 비교하려면 보충 자료를 참조하라. 실제 비디오 비교는 당사의 보충 비디오를 참조하라. 앞서 언급한 측정 지표에 대한 코드를 구현하고 그 유효성을 확인하기 위해 여러개의 공개된 비디오 결과에 대해 온전한 상태 점검을 실시하고 지속적으로 보고된 점수를 획득했다.

우리의 방법이 보여진 12개의 비디오 중 대다수를 위한 최고의 성능을 보여 준다는 것을 알 수 있다. 특히 이 방법은 자르기 비율이 1.0이며 모든 비디오에 대해 왜곡 값을 1.0에 가깝게 유지한다. 우리의 방법은 대다수의 비디오에서 가장 높은 안정성 점수와 몇가지 비슷한 결과를 보여 준다.


![Fig10](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig10.png)



### 4.2 Visual comparison
질량적 평가를 위해, SOTA와 시각적으로 비교한다. Fig. 9는 [Liu et al. 2013b], [Grundmann et al. 2011], [Liu et al. 2016] 방법을 사용하여 세 개의 비디오에서 안정화된 프레임의 예를 보여준다. 해당 입력 프레임과 비교하여 세 가지 기준 모두 어느 정도의 cropping 및 enlargement (확대) effect를 가지고 있는 반면, DIFRINT는 그렇지 않다. 마치 카메라가 비디오 프레임을 안정화하기 위해 움직인 것처럼 보이며, 그것등를 잘라내는 대신 보이지 않는 영역을 만들어낸다. 보이지 않는 지역의 생성을 Fig. 14에서 자세히 볼 수 있다. deep iterative frame interpolation의 결과로, 해당 입력 프레임에서 볼 수 없는 영상 경계의 내용을 생성할 수 있다.

![Fig11](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig11.png)
![Fig12](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig12.png)

안정화를 위해  frame interpolation 기법을 사용하기 때문에, 전체 이미지 외곡이 상당히 낮다. DIFRINT를 Fig. 10의 SOTA [Wang et al. 2018]와 비교한다. DIFRINT는 왜곡이 적고 더 많은 원본 내용을 포함하는 것을 볼 수 있다.  

ablation study로서, perceptual loss의 영향을 ![equ](http://latex.codecogs.com/svg.latex?l^{1})-loss function만을 사용한 것과 비교하였다. perceptual loss을 사용하면 stabilization metrics (cropping ratio, distortion value, and stability score) 측면에서 약간의 (한계적인) 개선이 보였다. 대조적으로 추가적인 perceptual loss를 사용하여 Fig. 15와 같이 시각적 품질이 개선되었음을 보여준다. 반복적인 reconstruction으로 인해, ![equ](http://latex.codecogs.com/svg.latex?l^{1})-loss만 사용하면 경계 근처에 흐릿하고 어두운 아티팩트가 발생한다. 또한 정량적 ablation study를 다양한 반복 횟수와 skip parameter에 대해 수행한다.


### 4.3 User study
상업적 알고리즘에 대해 평가하기 위해 Adobe Premiere Pro CC 2017에서 warp stabilizer, YouTube stabilizer [Grundmann et al. 2011]와 비교한다.  

![Fig13](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig13.png)

먼저 [Xu et al. 2018]에서 수행한 category-wise 비교를 Fig. 11에 나타난 6개의 카테고리로 수행한다: _Regular,_ _Crowd,_ _Parallax,_ _Quick_ _Rotation,_ Running,_ and _Zooming_. 6개의 카테고리에는 많은 이동하는 사람, 큰 시차, 빠른 카메라 회전, 달리기로 인한 심한 흔들림, 확대 축소로 인한 흔들림을 포함하는 어려운 시나리오가 포함된다. DIFRINT가 전반적으로 더 나은 성능을 보여준다. 특히 Running 카테고리와 같이 심한 흔들림이 있는 까다로운 장면의 경우 상업 알고리즘은 연속적인 missing view를 보완하기 위해 여백을 광범위하게 cropping 하는 것을 보여준다. 이와는 대조적으로 Fig. 12에서 보듯이, DIFRINT는 중요한 내용을 유지한다. 게다가, 상업적인 방법이 특히 심하게 흔들리는 비디오 부분에서 약간의 흔들리는 아티팩트와 왜곡을 유발하는 것을 알 수 있다.  
user study는 42명의 참가자들을 대상으로 Adobe Premiere, Robust L1 (YouTube)와 DIFIRNT의 선호도 테스트를 실시했다. 선호도 결과는 Fig. 13과 같다. DIFRINT가 대다수의 선호도를 가지고 있다는 것을 관찰할 수 있다.

![Fig14](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig14.png)
![Fig15](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig15.png)

## 5 Limitations & Discussion
DIFIRNT의 장점은 사용자가 원하는 대로 방복 횟수와 skip parameter를 조정할 수 있다는 것이다 (획일적인 방법 대신 조작의 자유 제공). 그러나 모든 모션 유형에 linear interpolation을 수행한다. 가능한 추후 연구에는 다른 비디오 세그먼트의 모션 유형을 동적으로 결정하고 그에 따라 안정화를 적용하는 것이다. 간단한 방법은 프레임 간 호모그래피를 통해 카메라 흔들림의 양을 측정하고 불안정성 정도에 따라 각 세그먼트에 다른 수의 안정화 반복을 적용하는 것이다.  

접근 방식의 또 다른 제한 사항은 심하게 흔들리는 동영상의 이미지의 경계가 흐릿하게 될 수 있다는 것이다. 카메라가 심하게 흔들리면 보이지 않는 큰 영역이 발생하기 때문에 이전 방법에서는 큰 영역을 잘라내는 경향이 있으며, 반면에 프레임 경계에 흐릿한 아티팩트가 나타난다. Fig. 16은 챌린지 비디오의 실패 사례로서 심각한 모호함의 예를 보여준다. 추후 연구는 여러 인접한 프레임을 명시적으로 이용하여 보이지 않는 큰 영역을 재구성하는 것이다. 문제는 메모리 문제를 처리하는 동시에 재구성 품질을 달성하는 것이다.

![Fig16](https://github.com/soraennon/VideoStabilization/blob/master/DIFRINT/figure/Fig16.png)


## 6 Conclusion
사용자가 개인 비디오에 비디오 안정화를 적용하기로 결정할 때, 사용자는 현재 어느 정도의 자르기와 확대를 고려해야 하며, 이것은 콘텐츠의 손실과 원하지 않는 확대 효과로 이어진다. 불필요한 영향 없이 개인 동영상을 안정시키는 것을 목표로 하는 방법을 제안한다. 또한 DIFRINT의 경량 프레임워크는 실시간에 가까운 컴퓨터 속도를 제공한다. 이 방법은 반복 프레임 보간을 통한 비디오 안정화에 대한 감독되지 않은 깊은 학습 접근 방식으로, 시각적 왜곡이 적은 전체 프레임 비디오를 안정화하는 것이다. 제안된 방법에서 우리는 반복 프레임 보간법이 비디오 안정화 작업에 유용한 것으로 간주되기를 바란다.
