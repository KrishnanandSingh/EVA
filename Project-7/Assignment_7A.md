# Assignment 7A

Receptive Field calculation for [GoogLeNet incarnation of the Inception architecture](https://arxiv.org/pdf/1409.4842.pdf)

## Considerations

- 1x1 layer are not shown here as they don't effect the RF.
- Only 5x5 layer is considered in inception blocks as they are contributing most to RF.

## Calculation Table

|Layer| K|P |S |N_in|N_out|R_out|J_in|
|-----|--|--|--|---|--|--|--|
|convolution |7|3|2|224|112|7|2|
|max pool|3|1|2|112|56|15|4|
|convolution|3|2|1|56|58|31|8|
|max pool|3|1|2|58|29|47|8|
|inception (3a)|5|2|1|29|29|111|16|
|inception (3b)|5|2|1|29|29|175|16|
|max pool|3|1|2|29|15|207|16|
|inception (4a)|5|2|1|15|15|335|32|
|inception (4b)|5|2|1|15|15|463|32|
|inception (4c)|5|2|1|15|15|591|32|
|inception (4d)|5|2|1|15|15|719|32|
|inception (4e)|5|2|1|15|15|847|32|
|max pool|3|2|2|15|9|911|32|
|inception (5a)|5|1|1|9|7|1167|64|
|inception (5b)|5|1|1|7|5|1423|64|
|avg pool|7|1|1|5|1|1807|64|
