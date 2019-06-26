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
![09](https://i.imgur.com/FxkFsIw.png) original image
![09_r1](https://i.imgur.com/SuHQVKj.png) DnCNN(depth=7) Residual
![09_r2](https://i.imgur.com/nuSUDln.png) Model Residual

### 3. 문제
- 수렴이 안됨 -> loss? network? dataset? learning rate?

### 4. 원인분석
- Network를 더 deep 하게
- FFT Shift
- loss function

### 5. To-Do
- [ ] Gabor filtering 더 알아보기
- [x] FFT-IFFT 구조로 테스트해보기 -> 되면 loss/learning rate 변경해서 다시 실험
- [ ] DnCNN depth=7 외에도 depth 12, 17 다양하게 실험

### 6. Reference

- Noise2Noise: [link](https://arxiv.org/pdf/1803.04189.pdf)
- Noise2Void: [link](http://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf)

## 190626

### Experiment
DnCNN - FFT - fftshift - 1 filter - ifftshift - iFFT 구조로 테스트함

저역통과필터를 만들어서 적용함 (d=100)

### Result
![09](https://i.imgur.com/FxkFsIw.png) original image
![09_residual](https://i.imgur.com/alkC8Pg.png) DnCNN(depth=17) residual image
![09_fft](https://i.imgur.com/mpm8YpR.png) result image

- 원하는 texture들은 texture들로 인식하지 않고, 특정 noise pattern 들이 texture로 인식됨.
- gabor filtering으로 하는것이 더 나아보임.
