## What are Channels and Kernels (according to EVA)?

Channels are bundle of similar features. Each channel contain their specific features which when combined together produce the whole experience. e.g. RGB channels have specific colors but together they create the colourful images.

In simplistic terms a Kernel is something that helps us to extract desired *features* from an input. Choice of a Kernel depends upon the *problem* we are trying to solve. To illustrate this, lets take an example.

**Problem:** We want to find circle and lines from the below input image.
![InputImage.png](https://user-images.githubusercontent.com/6766061/56986077-7a796f80-6ba7-11e9-978f-98c0a235491f.png)

**Features:**
So circle and line are the distinct features we want to extract.

**Choosing kernels:**
For this we will choose two kernels:
- Lets assume our first kernel (lets call this Circular kernel) is a special kernel that let pass only circles.
- Our second kernel(lets call this Line kernel) is another special kernel that let pass only lines.

**Extracting features:**
We will pass both kernels one by one over the input image. Circular kernel allows only circles so we will get a circle as the output. And similarly Line kernel produces lines and blocks everything else.

![Kernel](https://user-images.githubusercontent.com/6766061/56986089-81a07d80-6ba7-11e9-9d84-d448cabdcac7.png)


## Why should we only (well mostly) use 3x3 Kernels?

We don't use (even x even) kernels as there is no line of symmetry so we have to use (odd x odd) kernels. Now we can use any higher (odd x odd) kernel but it comes with computation cost. The smaller the kernel, the easier it is on the GPU. As we cannot use (1 x 1) kernel to detect anything our bet is on the second smallest odd number which is (3 x 3). There are two more reasons to use (3 x 3):

    1. GPUs are tuned to work better with (3 x 3)
    2. Any higher (odd x odd) kernel can be achieved by applying (3 x 3) multiple times


## Calculating number of 3x3 convolutions to reach 1x1 on a 199x199

3x3 convolution decreases the size by 2.

To reach 1x1 from 199x199 will take *(199-1)/2 = 99* 3x3 convolutions.

```txt
 (199x199)
    ||
    \/
 (197x197)
    ||
    \/
 (195x195)
    ||
    \/
 (193x193)
    ||
    \/
 (191x191)
    ||
    \/
 (189x189)
    ||
    \/
 (187x187)
    ||
    \/
 (185x185)
    ||
    \/
 (183x183)
    ||
    \/
 (181x181)
    ||
    \/
 (179x179)
    ||
    \/
 (177x177)
    ||
    \/
 (175x175)
    ||
    \/
 (173x173)
    ||
    \/
 (171x171)
    ||
    \/
 (169x169)
    ||
    \/
 (167x167)
    ||
    \/
 (165x165)
    ||
    \/
 (163x163)
    ||
    \/
 (161x161)
    ||
    \/
 (159x159)
    ||
    \/
 (157x157)
    ||
    \/
 (155x155)
    ||
    \/
 (153x153)
    ||
    \/
 (151x151)
    ||
    \/
 (149x149)
    ||
    \/
 (147x147)
    ||
    \/
 (145x145)
    ||
    \/
 (143x143)
    ||
    \/
 (141x141)
    ||
    \/
 (139x139)
    ||
    \/
 (137x137)
    ||
    \/
 (135x135)
    ||
    \/
 (133x133)
    ||
    \/
 (131x131)
    ||
    \/
 (129x129)
    ||
    \/
 (127x127)
    ||
    \/
 (125x125)
    ||
    \/
 (123x123)
    ||
    \/
 (121x121)
    ||
    \/
 (119x119)
    ||
    \/
 (117x117)
    ||
    \/
 (115x115)
    ||
    \/
 (113x113)
    ||
    \/
 (111x111)
    ||
    \/
 (109x109)
    ||
    \/
 (107x107)
    ||
    \/
 (105x105)
    ||
    \/
 (103x103)
    ||
    \/
 (101x101)
    ||
    \/
  (99x99)
    ||
    \/
  (97x97)
    ||
    \/
  (95x95)
    ||
    \/
  (93x93)
    ||
    \/
  (91x91)
    ||
    \/
  (89x89)
    ||
    \/
  (87x87)
    ||
    \/
  (85x85)
    ||
    \/
  (83x83)
    ||
    \/
  (81x81)
    ||
    \/
  (79x79)
    ||
    \/
  (77x77)
    ||
    \/
  (75x75)
    ||
    \/
  (73x73)
    ||
    \/
  (71x71)
    ||
    \/
  (69x69)
    ||
    \/
  (67x67)
    ||
    \/
  (65x65)
    ||
    \/
  (63x63)
    ||
    \/
  (61x61)
    ||
    \/
  (59x59)
    ||
    \/
  (57x57)
    ||
    \/
  (55x55)
    ||
    \/
  (53x53)
    ||
    \/
  (51x51)
    ||
    \/
  (49x49)
    ||
    \/
  (47x47)
    ||
    \/
  (45x45)
    ||
    \/
  (43x43)
    ||
    \/
  (41x41)
    ||
    \/
  (39x39)
    ||
    \/
  (37x37)
    ||
    \/
  (35x35)
    ||
    \/
  (33x33)
    ||
    \/
  (31x31)
    ||
    \/
  (29x29)
    ||
    \/
  (27x27)
    ||
    \/
  (25x25)
    ||
    \/
  (23x23)
    ||
    \/
  (21x21)
    ||
    \/
  (19x19)
    ||
    \/
  (17x17)
    ||
    \/
  (15x15)
    ||
    \/
  (13x13)
    ||
    \/
  (11x11)
    ||
    \/
  (9x9)
    ||
    \/
  (7x7)
    ||
    \/
  (5x5)
    ||
    \/
  (3x3)
    ||
    \/
  (1x1)
```
