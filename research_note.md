# Image Denoising with Structure Extraction from Residual image

## 190624

### 1. Model
- FFT
- Conv2d
- BatchNorm2d
- ReLU
- Conv2d
- BatchNorm2d
- ReLU
- Conv2d
- IFFT

### 2. Result
![09_r1](https://i.imgur.com/SuHQVKj.png) DnCNN(depth=7) Residual
![09_r2](https://i.imgur.com/nuSUDln.png) Model Residual

### 3. 문제
1. 수렴이 안됨 -> loss? network? dataset? learning rate?

### 4. 원인분석
1. Network를 더 deep 하게
2. FFT Shift
3. loss function

### 5. To-Do
- [ ] Gabor filtering 더 알아보기
- [ ] FFT-IFFT 구조로 테스트해보기 -> 되면 loss/learning rate 변경해서 다시 실험
- [ ] DnCNN depth=7 외에도 depth 12, 17 다양하게 실험
