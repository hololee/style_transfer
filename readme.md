## Style transfer - pytorch   
This project is practice to learn pytorch.  
Followed pytorch tutorial and implement below paper.  

[`A Neural Algorithm of Artistic Style`](https://arxiv.org/pdf/1508.06576v2.pdf)
  
  
### Result.

Type|base|content|style|output
|:----:|:----:|:----:|:----:|:----:|
LBFGS|![](https://github.com/hololee/style_transfer/blob/master/OUTPUT/LBFGS/input.png?raw=true)|![](https://github.com/hololee/style_transfer/blob/master/OUTPUT/LBFGS/content.png?raw=true)|![](https://github.com/hololee/style_transfer/blob/master/OUTPUT/LBFGS/style.png?raw=true)|![](https://github.com/hololee/style_transfer/blob/master/OUTPUT/LBFGS/output.png?raw=true)
SGD|![](https://github.com/hololee/style_transfer/blob/master/OUTPUT/SGD/input.png?raw=true)|![](https://github.com/hololee/style_transfer/blob/master/OUTPUT/SGD/input.png?raw=true)|![](https://github.com/hololee/style_transfer/blob/master/images/starry_night.jpg?raw=true)|![](https://github.com/hololee/style_transfer/blob/master/OUTPUT/SGD/output.png?raw=true)

- In SGD case, content image used as base.
