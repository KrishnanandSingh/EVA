# Assignment 7A

Receptive Field calculation for [GoogLeNet incarnation of the Inception architecture](https://arxiv.org/pdf/1409.4842.pdf)

## Considerations

- 1x1 layer are not shown here as they don't effect the RF.
- Only 5x5 layer is considered in inception blocks as they are contributing most to RF.

## Calculation Table

|Layer         |K|P|S|N_in|N_out|R_out|J_in|
|--------------|-|-|-|----|-----|-----|----|
|convolution   |7|3|2|224 |112  |7    |2   |
|max pool      |3|1|2|112 |56   |15   |4   |
|convolution   |3|1|1|56  |56   |31   |8   |
|max pool      |3|1|2|56  |28   |47   |8   |
|inception (3a)|5|2|1|28  |28   |111  |16  |
|inception (3b)|5|2|1|28  |28   |175  |16  |
|max pool      |3|1|2|28  |14   |207  |16  |
|inception (4a)|5|2|1|14  |14   |335  |32  |
|inception (4b)|5|2|1|14  |14   |463  |32  |
|inception (4c)|5|2|1|14  |14   |591  |32  |
|inception (4d)|5|2|1|14  |14   |719  |32  |
|inception (4e)|5|2|1|14  |14   |847  |32  |
|max pool      |3|1|2|14  |7    |911  |32  |
|inception (5a)|5|2|1|7   |7    |1167 |64  |
|inception (5b)|5|2|1|7   |7    |1423 |64  |
|avg pool      |7|0|1|7   |1    |1807 |64  |

## Notes

Receptive field is 1807 and not 224. I do not understand how they have calculated in the paper.
