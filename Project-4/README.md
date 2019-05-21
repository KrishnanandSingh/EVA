# Architectural Basics

EVA: Assignment 4

**Assumption:** Object size is equal to the image size.

## How many layers

The focus should be on reaching the full image/object size. We would add as many layers as required to reach the receptive field of the image. The final layer should be such that it has seen the whole image/object. Number of layers would vary depending on the max pool and the convolutions layers used.

## MaxPooling

Our objective is that the receptive field of the final layer should be equal to the input image. To reach this objective we can use only convolutions but this approach will create a large number of parameters and a lot of convolutions. In max pool we let forward only those values which stand out. So by adding a maxpooling layer we don't loose relevant information and the receptive field is also reduced(half if 2x2 is used). This helps us to reach our objective faster. That being said, max pool should not be used closer to the output layer as we don't want to loose *any* information at this level. And also we want to do some convolutions before doing a max pool so it should be used at  appropriate positions.

## 1x1 Convolutions

Think of this as a channel mixer. After we have done a few convolutions(lets say our filters have starting to recognize textures), now instead of reconvoluting, existing channels are used to create more complex channels. Also this helps in reducing the number of parameters.

## 3x3 Convolutions

The smaller the kernel the easier the convolution. Even size kernels are not useful as they don't have a line of symmetry and we cannot extract any pattern with 1x1. So 3x3 is the smallest kernel we have that is useful. Also GPU manufactureres like Nvidia have optimized the GPU for this kernel. Convolution with any higher kernel can be achieved by using combination of 3x3 kernels. (e.g. 5x5 is same as 3x3 followed by 3x3)

## Receptive Field

The number of pixels a block in output has convoluted on directly is called local receptive field. Suppose we have 5x5 input image and we add a 3x3 layer. After we convolve we get an output of 3x3 and each of the output pixel has seen 3x3 pixels of the input image. If we add another 3x3 layer on top of it, after convolution we will get only 1 pixel as the output. This pixel has seen only 3x3 pixels from previous layer so *local recptive field* is 3x3 but the *global recptive field* is 5x5 as the 3x3 pixels from previous layer have already convolved on 5x5 image.

To get a good prediction we would need the network to have seen the whole image(assuming object size is equal to the image size).

## SoftMax


## Learning Rate


## Kernels and how do we decide the number of kernels?


## Batch Normalization


## Image Normalization


## Position of MaxPooling


## Concept of Transition Layers


## Position of Transition Layer


## Number of Epochs and when to increase them


## DropOut


## The distance of Batch Normalization from Prediction


## When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)

## When to add validation checks

TODO:
```py
model.fit(X_train, Y_train, batch_size=32, nb_epoch=6, verbose=1, validation_data=(X_test, Y_test))

## When do we introduce DropOut, or when do we know we have some overfitting

When we observe that our network is doing very well on the training data but doing poor on the validation data. In other words when the training accuracy is high but the validation accuracy is low we know that the model is overfitting. At this stage we introduce Dropout.

## The distance of MaxPooling from Prediction

Max pooling helps us reaching the receptive pool faster but it should not be used closer to the prediction layer. As by using max pool we loose all values other than the max ones. This may not be an issue when our global receptive field is small but when we go up the network the values become more relevant than before. Before the prediction layer we don't want to loose *any* information so we don't do max pools before 3 layers.

```

## How do we know our network is not going well, comparatively, very early

Always use test data as validation data. Now after each epoch we get to see the actual validation accuracy. If the validation accuracy for the first epoch is worse than the previous network the current one is not a better network.

## Batch Size, and effects of batch size

### Batch size

Batch size is the number of training data the network sees paralelly. Suppose that if we have 1000 training data and 100 is the batch size, then the network will fit(calculate loss) 100 training data at a time and adjust itself(back propogation). Which means that it would have to repeat this process 10 times to be able to fit on all training data.

### Effects of batch size

If we have unlimited GPU, the greater the batch size the faster the training. But practically since we have limited GPU as we increase the batch size training speed increases upto a certain batch size and then tends to slow down ultimately reaching at OOM.

With a larger batch size the network gets to see more training data at a time so it can adjust better. But note that with a smaller batch size the network gets more oppurtunities to learn. So if we have a higher batch size we can use higher learning rate.


## LR schedule and concept behind it

When the network is approaching global minima the LR should be low. If the learning rate is high it takes larger steps and wobbles on the edges. It would have been dropped into the minima if it would have taken small steps at this phase. So we schedule the learning rate to be higher when the network has just started learning and to small values when it is about to reach the target global minima.

## Adam vs SGD

There is no clear consensus on which one is better. Both work pretty well.
