### project
참조 라이브러리

FAST-SNN https://github.com/yangfan-hu/Fast-SNN

ESVAE https://github.com/qgzhan/esvae

### base, utils,data, metrics는 ESVAE 참조
### origin은 ESVAE의 ANN_VAE 사용 
### train은 입맛에 맞게 수정 완료 

### issue

1. Summarywriter에서 Pillow 버전 문제

2. metrics에서는 inception만 됨 ( FID는 HTTPS 404 error 발생 )

### 07/14 업데이트 
#### Spiking_IF 은 IF 뉴런과 Last_Spiking에 IF 뉴런 삽입한 형태
#### Spiking_LIF 은 LIF 뉴런과 Last_Spiking에 LIF 뉴런 삽입한 형태
#### Quant_train --> Snn_converting 하기 


### 07/26 업데이트
1. pip install ptflops==0.7.2.2 https://github.com/sovrasov/flops-counter.pytorch 
2. pip install syops https://github.com/iCGY96/syops-counter 