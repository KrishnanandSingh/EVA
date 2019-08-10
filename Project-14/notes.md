
# Suggestions

Quoting Rohan

1. [ ] Look at the log, the model I shared is overfitting.. how do we solve overfitting?
2. [ ] Remember AutoImg from Google? Anyone added that? No rotation in my code, any one added that?
3. [ ] Moved to Tensorflow BatchNorm?
4. [ ] Removed strides?
5. [ ] Do you think this is the only model? What if we go slightly deeper or wider?
6. [ ] Performed Augmentations and stored the images?
7. [ ] Tried any other LR?
8. [ ] Did you run the same core at least for one more epoch? Code given is already at 93% after 24. What happens if we perform gradual slowdown after 24 epochs?
9. [ ] Not able to add CutOut to code shared? What about adding the model I shared to your own assignment 6-7-8-9-10 code?
10. [ ] Did you notice we are using same mean std values for test and train images? What did we discuss on that?


## 3. Use batchnorm from tensorflow

### Speed Comparison of using keras library directly vs tf.keras

- Without batch norm each epoch in tf.keras was 2 seconds faster than in keras
- with batch norm it became ~5seconds faster
- In Keras 4 seconds extra due to BN
- In tf.keras 1-2 seconds extra

---

- [TFBatchNorm.ipynb](https://colab.research.google.com/drive/1Z4lThL2A0xElxOIytMUdy2JpZm4e2TaR)
- [KerasBatchNorm.ipynb](https://colab.research.google.com/drive/1Z52MFqNCXMPt5M9jm8KGUB93V_1EoV4c)

## 7,8 Increase max LR

Experiments with LR and findings:

Original config

```
LR = 0.4
EPOCHS = 24

epoch: 24
train loss: 0.02025775463104248
train acc: 0.99556
val loss: 0.2535373443603516
val acc: 0.931
time: 858.5140795707703
```

Increasing epochs did increase the accuracy but only a little
```
LR = 0.6
EPOCHS = 30

epoch: 30
train loss: 0.008743435363769531
train acc: 0.99818
val loss: 0.2722068435668945
val acc: 0.9299
time: 1049.8495931625366
```

Better config
```
LR = 0.8
EPOCHS = 20

epoch: 20
train loss: 0.026008583660125734
train acc: 0.9931
val loss: 0.25837910537719727
val acc: 0.9297
time: 705.1857738494873
```

Decreasing epoch with same LR

```
LR = 0.8
EPOCHS = 15

epoch: 15
train loss: 0.05811372247695923
train acc: 0.9827
val loss: 0.25160755538940427
val acc: 0.9203
time: 542.6324870586395
```

Increasing LR with same epochs

```
LR = 1.0
EPOCHS = 15

epoch: 15
train loss: 0.055607344875335696
train acc: 0.98306
val loss: 0.25801772994995115
val acc: 0.9227
time: 532.2575883865356
```

Increasing momentum

```
MOMENTUM=0.93
LR=1.0
EPOCHS=15

epoch: 15
train loss: 0.050490148944854735
train acc: 0.98456
val loss: 0.262967985534668
val acc: 0.9201
time: 550.328127861023
```

Increasing to 94 decreased accuracy

```
MOMENTUM=0.92
LR=1.0
EPOCHS=15

epoch: 15
train loss: 0.04895516384124756
train acc: 0.98518
val loss: 0.26813325500488283
val acc: 0.9223
time: 534.1940670013428
```

Increasing weight decay to 5e-3 didn't have much effect on accuracy but decreased training time by ~16 seconds

```
MOMENTUM=0.92
LR=1.0
EPOCHS=15
WEIGHT_DECAY:5e-3

epoch: 15
train loss: 0.05117958498001099
train acc: 0.98462
val loss: 0.2687091995239258
val acc: 0.9193
time: 518.0960686206818
```

Increasing LR to 1.5

```
LR = 1.5
EPOCHS = 15

epoch: 15
train loss: 0.05141395936965942
train acc: 0.98376
val loss: 0.26824850120544436
val acc: 0.9211
time: 537.7505202293396
```

```
LR = 1.5
EPOCHS = 20

epoch: 20
train loss: 0.024087078032493592
train acc: 0.9931
val loss: 0.26868133697509766
val acc: 0.9275
time: 726.5625402927399
```

## momentum should also be varied according to the LR

src: https://github.com/titu1994/keras-one-cycle

## 2,6. Apply autoaugment policies for cifar

|                |     Operation 1     | Operation 2      |
|----------------|---------------------|------------------|
|Sub-policy 0    |(Invert,0.1,7)       | (Contrast,0.2,6)
|Sub-policy 1    |(Rotate,0.7,2)       | (TranslateX,0.3,9)
|Sub-policy 2    |(Sharpness,0.8,1)    | (Sharpness,0.9,3)
|Sub-policy 3    |(ShearY,0.5,8)       | (TranslateY,0.7,9)
|Sub-policy 4    |(AutoContrast,0.5,8) | (Equalize,0.9,2)
|Sub-policy 5    |(ShearY,0.2,7)       | (Posterize,0.3,7)
|Sub-policy 6    |(Color,0.4,3)        | (Brightness,0.6,7)
|Sub-policy 7    |(Sharpness,0.3,9)    | (Brightness,0.7,9)
|Sub-policy 8    |(Equalize,0.6,5)     | (Equalize,0.5,1)
|Sub-policy 9    |(Contrast,0.6,7)     | (Sharpness,0.6,5)
|Sub-policy 10   |(Color,0.7,7)        | (TranslateX,0.5,8)
|Sub-policy 11   |(Equalize,0.3,7)     | (AutoContrast,0.4,8)
|Sub-policy 12   |(TranslateY,0.4,3)   | (Sharpness,0.2,6)
|Sub-policy 13   |(Brightness,0.9,6)   | (Color,0.2,8)
|Sub-policy 14   |(Solarize,0.5,2)     | (Invert,0.0,3)
|Sub-policy 15   |(Equalize,0.2,0)     | (AutoContrast,0.6,0)
|Sub-policy 16   |(Equalize,0.2,8)     | (Equalize,0.6,4)
|Sub-policy 17   |(Color,0.9,9)        | (Equalize,0.6,6)
|Sub-policy 18   |(AutoContrast,0.8,4) | (Solarize,0.2,8)
|Sub-policy 19   |(Brightness,0.1,3)   | (Color,0.7,0)
|Sub-policy 20   |(Solarize,0.4,5)     | (AutoContrast,0.9,3)
|Sub-policy 21   |(TranslateY,0.9,9)   | (TranslateY,0.7,9)
|Sub-policy 22   |(AutoContrast,0.9,2) | (Solarize,0.8,3)
|Sub-policy 23   |(Equalize,0.8,8)     | (Invert,0.1,3)
|Sub-policy 24   |(TranslateY,0.7,9)   | (AutoContrast,0.9,1)

AutoAugment policy found on reduced CIFAR-10.

src: [Autoaugment](https://arxiv.org/pdf/1805.09501.pdf)
