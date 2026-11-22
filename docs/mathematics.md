# From a point to a parameter update

## Forward propagation

One example occupies one row. For a batch of $B$ points with $d$ features, $A_0=X$ has shape $(B,d)$. A dense layer has a weight matrix $W$ of shape $(d,h)$ and a bias vector $b$ of length $h$. Matrix multiplication takes the dot product of each example with each column of weights. Broadcasting adds the same bias to each row.

$$Z_i=A_{i-1}W_i+b_i,\qquad A_i=f(Z_i).$$

ReLU takes $\max(0,z)$ with derivative 1 for positive inputs and 0 for negative inputs. The implementation chooses 0 at the kink. Tanh maps real numbers into $(-1,1)$ with derivative $1-\tanh^2(z)$. Without a nonlinear activation, composing dense layers still gives a single affine map. The output layer returns raw class scores called logits.

## A complete example

Take a single input $x=[1,2]$, one hidden ReLU unit with weights $[1,-0.25]^T$ and zero bias. Its preactivation is $1(1)+2(-0.25)=0.5$, so its activation is $0.5$. Output weights $[2,0]$ and zero biases yield logits $[1,0]$. Suppose the correct label is class 1 (the second column).

Softmax gives each class probability $p_j=e^{z_j-m}/\sum_k e^{z_k-m}$, where $m=\max_k z_k$. Subtracting the maximum keeps the largest exponential at 1. Here the shifted scores are $[0,-1]$ and probabilities are $[0.731059,0.268941]$. The loss is $-\log(0.268941)=1.313262$. In code we evaluate the equivalent expression directly in log space, so an exponential underflow for the target class still gives a finite loss for ordinary finite logit gaps.

$$L=\frac1B\sum_r\left[\log\sum_k e^{z_{rk}-m_r}-(z_{r,y_r}-m_r)\right].$$

## Backpropagation

The chain rule multiplies local derivatives along each path from a weight to the loss and adds contributions from different paths. Differentiating log-sum-exp gives softmax. Differentiating the subtracted target logit gives a one-hot vector. The batch-mean output derivative is therefore $D=(P-Y)/B$.

$$\nabla_W L=A^T D,\qquad \nabla_b L=\sum_{r}D_{r,:},\qquad D_{\rm prev}=(DW^T)\odot f'(Z_{\rm prev}).$$

In the example $D=[0.731059,-0.731059]$. Multiplication by the hidden activation gives output weight gradients $[0.365529,-0.365529]$. Output bias gradients equal $D$. Backpropagating through the output weights gives $2(0.731059)+0(-0.731059)=1.462118$. The hidden ReLU has slope 1, so the hidden bias gradient is $1.462118$ and the two input weight gradients are $[1.462118,2.924236]^T$. This is why the implementation caches layer inputs and waits until the backward pass finishes to update any weight.

## Updating and checking the weights

SGD subtracts learning rate times gradient. At rate $0.1$, the first output weight becomes $2-0.1(0.365529)=1.963447$. The second becomes $0.036553$. Each mini-batch provides a new estimate of the mean gradient.

Adam keeps an exponential mean of the gradients ($m$, decay 0.9) and their squares ($v$, decay 0.999), initialized at zero. At update $t$ divide each estimate by $1-\beta^t$ using its own decay, then subtract $\eta\hat m/(\sqrt{\hat v}+10^{-8})$.

To check a derivative, perturb just one scalar by $+h$ and $-h$ while holding all other parameters fixed. With $h=10^{-5}$, the centered difference $[L(w+h)-L(w-h)]/(2h)$ estimates its slope. The first output weight above should give approximately $0.365529$. The experiment checks every weight and bias of a tiny tanh network and reports the largest absolute discrepancy and $\lVert g-n\rVert_2/(\lVert g\rVert_2+\lVert n\rVert_2)$ for analytical and numerical gradient vectors $g,n$. Using float64 reduces cancellation error. A ReLU perturbation that crosses zero can disagree with a local derivative even for a correct implementation.
