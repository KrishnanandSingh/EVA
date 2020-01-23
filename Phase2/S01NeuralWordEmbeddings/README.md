# NeuralWordEmbeddings

We use pretrained embeddings when our train data is sparse. When we have ample train data, learning our own embedding outperforms pre-trained ones.
```
training_samples = 200
validation_samples = 10000
```
[0.7342179582595825, 0.57068]

![image](https://user-images.githubusercontent.com/6766061/73010144-dd96ea80-3e37-11ea-8529-a1381f7218b1.png)
![image](https://user-images.githubusercontent.com/6766061/73010176-edaeca00-3e37-11ea-8ee3-46194e25c40f.png)


Training the same model without pretrained word embeddings

![image](https://user-images.githubusercontent.com/6766061/73010849-31560380-3e39-11ea-9df3-e41c17f17ee9.png)
![image](https://user-images.githubusercontent.com/6766061/73010879-403cb600-3e39-11ea-8d3e-59fd02fce5e6.png)

We see that training with jointly learned embeddings performed poorly than the pre trained embeddings. Let's try training again with more train data.
```
training_samples = 8000
validation_samples = 10000
```
[0.6841680439170823, 0.87584]

![image](https://user-images.githubusercontent.com/6766061/73010312-36668300-3e38-11ea-91d3-028235646296.png)
![image](https://user-images.githubusercontent.com/6766061/73010318-39617380-3e38-11ea-8cbb-e0a6cdd164eb.png)


[0.6199846486377716, 0.78048]

![image](https://user-images.githubusercontent.com/6766061/73010350-4a11e980-3e38-11ea-8348-f54b3df4a38e.png)
![image](https://user-images.githubusercontent.com/6766061/73010360-4e3e0700-3e38-11ea-9fa4-c1e7f8e4cb19.png)
