# 딥러닝을 이용한 폭우 예측

## 멤버

김동욱,정보시스템학과,dongwook1214@gmail.com

이재흠,컴퓨터소프트웨어학과,rethinking21@gmail.com

## 동기&목표

지난 8월 서울에 기록적인 폭우가 내렸습니다.

이번 폭우의 1시간 최대 강수량은 141.5가 넘는 수치였습니다.

이는 평균적인 강수량인 30에 비해 4배가 넘는 수치입니다.

![Untitled](https://user-images.githubusercontent.com/69969001/204224371-f11b51e1-cae6-412c-853a-bcc8f5819a62.png)

이번 폭우로 인해 건물이 무너지거나 인명 피해가 나기도 했습니다.

반지하에 살던 사람은 집을 잃고 길거리에 나앉게 되었습니다.

[폭우 속 맨홀 빠져 실종됐던 여성 숨진 채 발견](https://www.ytn.co.kr/_ln/0103_202208120614289144)

전 서울에서 이번 폭우로 이 정도의 피해가 날거라고 예상한 지구는 한 곳도 없었다 합니다.

만약에 폭우를 예측할 수 있었다면 어땠을까요.

저희 팀은 딥러닝을 통해 폭우를 미리 예측하여 재난에 대응할 수 있는 가능성을 만들어보고자 합니다.

## 데이터 셋

데이터 전처리까지

[기상자료개방포털[기후통계분석:통계분석:강수량분석]](https://data.kma.go.kr/stcs/grnd/grndRnList.do)

[기상자료개방포털[기후통계분석:통계분석:다중지점통계]](https://data.kma.go.kr/climate/StatisticsDivision/selectStatisticsDivision.do?pgmNo=158)

날씨 데이터 셋(2010년대부터 멈춘듯)

[dataset: NOAA NCDC GHCN v2](https://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NCDC/.GHCN/.v2/IWMO+47108000+VALUE/)

날씨 관련 데이터셋(1시간 단위, 유료)

[Weather Archive Seoul - meteoblue](https://www.meteoblue.com/en/weather/archive/export/seoul_south-korea_1835848)

강우량 (10분단위, 서울 지역별)

[](https://data.seoul.go.kr/dataList/OA-1168/S/1/datasetView.do#)

날짜 입력시 표시(크롤링해야함)

[지역별상세관측자료(AWS)](https://www.weather.go.kr/weather/observation/aws_table_popup.jsp)

위 자료를 정리해놓은 사이트 (많이 느림) → **다운 완료(분, 시간, 일단위)**

[기상자료개방포털[데이터:기상관측:지상:방재기상관측(AWS):파일셋]](https://data.kma.go.kr/data/grnd/selectAwsRltmList.do?pgmNo=56&tabNo=1)

## 방법론

이번 프로젝트의 목표는 폭우를 예측하는 것입니다.

저희 팀은 현 날짜까지의 강수량 데이터를 모았고 이 데이터를 기반으로 미래의 데이터를 예측해보고자 합니다.

RNN은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 갖고있습니다. 

이는 곧 이전의 작업을 현재와 연결시킬 수 있고 이전의 데이터를 기반으로 상황에 맞는 결과를 낼 수 있다는 의미이기도 합니다.

따라서 RNN을 이용하면 이전의 강수량을 통해 현재의 강수량을 알 수 있습니다.

![Untitled](https://user-images.githubusercontent.com/69969001/204224715-3afb5ab4-b5d8-4bc0-ae82-e700652f076f.png)

                                                                   <RNN의 구조도>

RNN은 gradient 계산에서 곱셈 연산을 하는데 1보다 작은 값을 계속 곱하면서 결국 gradient는  0으로 수렴합니다.

이를 경사 소실 문제라고 합니다.

저희의 프로젝트에서는 약 6년 간의 데이터를 다루며 이는 경사 소실 문제로 이어질 수 있습니다.

이를 해결하기 위해 LSTM을 사용하기로 했습니다.

LSTM은 RNN의 한 종류이며 곱하기 연산이 아닌 더하기 연산을 사용합니다.

실제로 LSTM은 RNN 보다 훨씬 향상된 성능을 보여주며 이는 저희의 프로젝트에 적합해 보입니다.

reference.

[QANDA 머신 러닝 스터디](https://blog.mathpresso.com/mathpresso-%EB%A8%B8%EC%8B%A0-%EB%9F%AC%EB%8B%9D-%EC%8A%A4%ED%84%B0%EB%94%94-12-rnn-recurrent-neural-nerwork-1-b28968016ca9)

[Long Short-Term Memory (LSTM) 이해하기](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr)

## 데이터 처리 (encoding: ansi)

[데이터 처리 코드](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%E1%85%B5%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%91%E1%85%A9%E1%86%A8%E1%84%8B%E1%85%AE%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%207015363766494161b9b9034c7a84b5d3/%E1%84%83%E1%85%A6%E1%84%8B%E1%85%B5%E1%84%90%E1%85%A5%20%E1%84%8E%E1%85%A5%E1%84%85%E1%85%B5%20%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3%209751d88fa3f249f198c9f64d80b80127.md) 

- 1시간 단위 (날짜 누락 있음)
    
    [one_hour.zip](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%E1%85%B5%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%91%E1%85%A9%E1%86%A8%E1%84%8B%E1%85%AE%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%207015363766494161b9b9034c7a84b5d3/one_hour.zip)
    
- 1시간 단위 (날짜 누락 없음)
    
    [one_hour_2.zip](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%E1%85%B5%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%91%E1%85%A9%E1%86%A8%E1%84%8B%E1%85%AE%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%207015363766494161b9b9034c7a84b5d3/one_hour_2.zip)
    
- 3시간 단위
    
    [3hours.zip](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%E1%85%B5%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%91%E1%85%A9%E1%86%A8%E1%84%8B%E1%85%AE%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%207015363766494161b9b9034c7a84b5d3/3hours.zip)
    

## 학습시키기

일단 1일 단위로 나누어져 있는 데이터를 가져와 학습시켜보았다.

[1일 단위 학습 코드](%E1%84%83%E1%85%B5%E1%86%B8%E1%84%85%E1%85%A5%E1%84%82%E1%85%B5%E1%86%BC%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%8B%E1%85%B5%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%91%E1%85%A9%E1%86%A8%E1%84%8B%E1%85%AE%20%E1%84%8B%E1%85%A8%E1%84%8E%E1%85%B3%E1%86%A8%207015363766494161b9b9034c7a84b5d3/1%E1%84%8B%E1%85%B5%E1%86%AF%20%E1%84%83%E1%85%A1%E1%86%AB%E1%84%8B%E1%85%B1%20%E1%84%92%E1%85%A1%E1%86%A8%E1%84%89%E1%85%B3%E1%86%B8%20%E1%84%8F%E1%85%A9%E1%84%83%E1%85%B3%201695ac23a86748f28195660bfcdb7b22.md)

![Untitled](https://user-images.githubusercontent.com/69969001/204224402-df6fca70-5d79-42ff-814a-d3eac304b1d0.png)

다음과 같은 결과가 나왔다.

아마 0으로 돼있는 데이터가 많고 폭우가 내리는게 흔한일이 아니기에 예측을 잘 못하는 것 같다.

0과 데이터의 최댓 값 사이의 차이가 큰 것도 이유 중 하나 일거 같다.

하지만 정점의 x값 위치가 예측 데이터와 실제 측정 데이터가 비슷하기에 의미있는 데이터로 보인다.

![Untitled](https://user-images.githubusercontent.com/69969001/204224432-c044cb2c-7412-46f4-8e0b-08266fc9ca9a.png)

이건 3시간 단위의 데이터를 ephoc 50, batchSize 512로 돌려본거다.

![다운로드.png](https://user-images.githubusercontent.com/69969001/204224412-fc5d476a-a648-456b-8200-a86f75a3eebe.png)

ephoc 100, batchSize 32

최솟값과 최댓값의 차이가 적을 수록 예측 결과가 더 좋은 것 같다는 생각을 했다.

ephoc의 영향도 크게 보인다.

1시간 단위로 기록된 데이터로도 예측해봤다.

![Untitled](https://user-images.githubusercontent.com/69969001/204224421-baa7001d-4f17-44c8-ba00-a3db323c2653.png)

에포크 100, 배치 128

여기부터 다음주

[시계열 예측: LSTM 모델로 주가 예측하기](https://insightcampus.co.kr/2021/11/11/%EC%8B%9C%EA%B3%84%EC%97%B4-%EC%98%88%EC%B8%A1-lstm-%EB%AA%A8%EB%8D%B8%EB%A1%9C-%EC%A3%BC%EA%B0%80-%EC%98%88%EC%B8%A1%ED%95%98%EA%B8%B0/)

[How to Use Features in LSTM Networks for Time Series Forecasting - MachineLearningMastery.com](https://machinelearningmastery.com/use-features-lstm-networks-time-series-forecasting/)
