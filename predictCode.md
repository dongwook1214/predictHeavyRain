# 미래 예측 코드 & 코드 설명

*이 코드는 앞서 나온 학습코드에 이어지는 코드입니다.

학습데이터가 2021년까지 자료이므로 2021-8-24일 부터 예측하겠습니다.

```python
#2021-8-24일 부터 예측
x_data_input = dataset_total.values
x_data_input = x_data_input.reshape(-1, 1)
x_data_input = sc.transform(x_data_input)
x_data_input = x_data_input[61367-unitDevide:61367]
x_data = []
x_data_input.shape
```

x_data를 model에 들어갈 수 있는 형태로 바꾼뒤에 예측결과가 들어갈 predictResult 리스트를 만들어 줍니다.

이때, x_data의 형태는 (1,60,1)

```python
x_data.append(x_data_input[0:])
x_data = np.array(x_data)
x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
predictResult = []
```

“x_data중 가장 마지막을 골라” (이를 A라고 하겠습니다) 다음 데이터를 예측합니다.

예측 값이 나오면 predictResult에 추가한 뒤에

A 중 가장 첫번째 인덱스를 제외한 나머지를 x_data에 추가하고 예측된 값을 추가합니다.

 “np.reshape(x_data, (-1,60,1))” 다음 코드를 통해 다시 모델에 들어갈 수 있는 형태로 만들어줍니다

```python
#2023-8-24일까지 예측
for i in range(5840):
  temp = np.reshape(x_data[-1],(-1,60,1))
  prt = model.predict(temp)
  predictResult.append(prt[-1][0])
  print(prt[-1][0])
  x_data = np.append(x_data,x_data[-1][-unitDevide + 1 : ][0:])
  x_data = np.append(x_data,prt[-1][0])
  x_data = np.reshape(x_data, (-1,60,1))
```

다음으로 정규화를 풀어줍니다.

```python
temp = np.reshape(predictResult,(-1,1))
minMaxScalerRealse = sc.inverse_transform(temp)
```

결과를 출력합니다.

```python
plt.plot(minMaxScalerRealse, color = "blue", label = "predicted Precipitation")

plt.xticks(np.arange(0,len(minMaxScalerRealse),50))

plt.title("Precipitation forecast")

plt.xlabel("Time")

plt.ylabel("Precipitation")

plt.legend()

plt.show()
```