# One weight update, by hand

I use this example to connect the matrix operations in the code to a prediction I can check with a calculator. The experiment uses tanh, two hidden layers and mini-batches. Here I use one point and one ReLU unit so every derivative fits on the page. [Network code](../neuralforge/network.py)

## From coordinates to class scores

In `MLP.forward`, I arrange a batch as a matrix $A$ with $B$ rows and $d$ features. A weight matrix $W$ of shape $(d,h)$ contains one column per output unit. The product $AW$ takes the dot product of every row with every column. NumPy then adds the same $h$ biases to each row.

$$
Z=AW+b,\qquad A_{\mathrm{next}}=f(Z).
$$

Take $x=[1,2]$, hidden weights $[1,-0.25]^T$ and zero bias. The hidden unit receives $1(1)+2(-0.25)=0.5$. ReLU returns $\max(0,0.5)=0.5$. Now use output weights $[2,0]$ and zero output biases. The logits are $[1,0]$. I'll use class 1, the second column, as the correct answer.

The activation matters here. Without it, multiplying by a second weight matrix and adding another bias still gives an affine map of the input. Equal scores for two affine outputs define a straight line in two dimensions. The nonlinear hidden activations let the spiral model bend that boundary.

## From scores to loss

Softmax assigns probabilities proportional to the exponentials of the logits. Subtracting the same number from every logit leaves their probabilities unchanged. I subtract the maximum $m$ so the largest exponential is exactly 1.

$$
p_j=\frac{\exp(z_j-m)}{\sum_k\exp(z_k-m)}.
$$

For the example, the shifted scores are $[0,-1]$ and the probabilities are about $[0.731059,0.268941]$. The loss is $-\log(0.268941)\approx1.313262$. In `cross_entropy` I compute that loss directly from the shifted logits. That avoids taking the logarithm of a probability that could round to zero. Here $m_r$ is the largest logit in row $r$ and $y_r$ is the correct class index.

$$
L=\frac{1}{B}\sum_r\left[\log\sum_k e^{z_{rk}-m_r}-(z_{r,y_r}-m_r)\right].
$$

## From loss back to weights

The derivative of log-sum-exp is softmax. Subtracting the target score contributes $-1$ in the target column. If $Y$ holds one-hot labels and $P$ holds probabilities, the gradient with respect to the output logits is $D=(P-Y)/B$. In code I subtract 1 at the indexed labels so I can skip allocating the one-hot matrix.

For one dense layer with input $A$, the chain rule gives

$$
\nabla_W L=A^T D,\qquad \nabla_b L=\sum_r D_{r,:},\qquad \nabla_A L=DW^T.
$$

I can read the first equation one weight at a time. Each example contributes its input value times the derivative at the output that weight feeds. The matrix product adds those contributions over the batch. The loss derivative already includes the division by $B$.

| Quantity in the example | Calculation | Gradient, rounded |
| --- | --- | --- |
| Output logits and biases | Subtract target $[0,1]$ from the probabilities | $[0.731059,-0.731059]$ |
| Output weights | Multiply the logit gradient by the hidden activation $0.5$ | $[0.365529,-0.365529]$ |
| Hidden activation | Sum the contributions from both output weights | $2(p_0)+0(p_1-1)\approx1.462117$ |
| Hidden bias | Multiply by the ReLU slope at $0.5$ | $1.462117$ |
| Hidden weights | Multiply by each input coordinate | $[1.462117,2.924234]^T$ |

The table rounds intermediate values for display. In `MLP.loss_gradients` I walk the layers backward and multiply the incoming gradient elementwise by the activation slope. For tanh I reuse the cached activation $a$, since its slope is $1-a^2$. For ReLU I use 1 where the activation is positive and 0 elsewhere, including at the kink. All gradients use the weights from the same forward pass.

## Updating and checking a parameter

SGD subtracts learning rate times gradient. With a learning rate of $0.1$, the first output weight moves from $2$ to $2-0.1(0.365529)\approx1.963447$. The second moves from $0$ to approximately $0.036553$. Both changes favor the second class on this example.

For Adam I keep moving averages of the gradient and its square, starting at zero. I correct the early bias of each average by dividing it by $1-\beta^t$ using its own decay and the mini-batch update count $t$. The step I subtract is $\eta\hat m/(\sqrt{\hat v}+\epsilon)$. [Optimizer implementation](../neuralforge/training.py)

To check the first output-weight derivative independently, I can evaluate the loss at $2+h$ and $2-h$ while holding everything else fixed. With $h=10^{-5}$ the centered difference below should be close to $0.365529$.

$$
g_{\mathrm{numeric}}=\frac{L(w+h)-L(w-h)}{2h}.
$$

The automated check repeats this for every parameter in a small float64 network. It reports both the largest absolute difference and the relative vector error $\lVert g-n\rVert_2/(\lVert g\rVert_2+\lVert n\rVert_2)$, where $g$ is the analytical gradient and $n$ is the finite-difference estimate. I use a smooth tanh network for the reported check because a perturbation that straddles the ReLU kink samples two different slopes. [Finite-difference implementation](../neuralforge/check.py)
