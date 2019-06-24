# Image Denoising with Structure Extraction from Residual image

## 190624

### Model
- FFT
- Conv2d
- BatchNorm2d
- ReLU
- Conv2d
- BatchNorm2d
- ReLU
- Conv2d
- IFFT

### Result
![09_r1](https://i.imgur.com/SuHQVKj.png) DnCNN(depth=7) Residual
![09_r2](https://i.imgur.com/nuSUDln.png) Model Residual

### 문제
1. 수렴이 안됨 -> loss? network? dataset? learning rate?

### 원인분석
1. Network를 더 deep 하게
2. FFT Shift
3. loss function

### 할 일
1. Gabor filtering 더 알아보기
2. FFT-IFFT 구조로 테스트해보기 -> 되면 loss/learning rate 변경해서 다시 실험
