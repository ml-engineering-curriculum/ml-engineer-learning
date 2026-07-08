# 03 — Derivatives, gradients, and the chain rule

> **Reading time:** ~40 minutes.
> **Prerequisite chapters:** 02.

## Motivation

Training a model is, almost without exception, the act of repeatedly
asking: *how should I change the parameters to make the loss smaller?* The
mathematical object that answers that question is the **gradient**. If you
understand what a gradient is and what the chain rule does, you understand
the engine that drives every modern ML system, even if you never write
the backward pass yourself.

This chapter is the calculus equivalent of chapter 02: the working
engineer's tour, not a textbook.

## Derivatives: the one-variable case

For a function $f: \mathbb{R} \to \mathbb{R}$, the derivative at a point
$x$ is

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}.$$

Operationally, $f'(x)$ is the slope of the tangent line at $x$ — the
local linear approximation. If $f'(x) > 0$, $f$ is increasing at $x$; if
$f'(x) < 0$, it is decreasing. If $f'(x) = 0$, you are at a stationary
point — a candidate minimum, maximum, or saddle.

You should be able to recite the derivatives of the half-dozen
functions that appear constantly in ML:

| $f(x)$ | $f'(x)$ | Where you see it |
|---|---|---|
| $x^2$ | $2x$ | squared-error loss |
| $e^x$ | $e^x$ | softmax, sigmoid |
| $\log x$ | $1/x$ | cross-entropy, log-likelihood |
| $\sigma(x) = 1/(1+e^{-x})$ | $\sigma(x)(1 - \sigma(x))$ | logistic regression |
| $\text{ReLU}(x) = \max(0, x)$ | $1$ if $x > 0$, else $0$ | neural network activations |
| constant $c$ | $0$ | (biases differentiate cleanly) |

You don't have to derive these from scratch every time, but you should
recognise them when they appear in a gradient calculation.

## Partial derivatives and the gradient

Real ML models depend on many parameters, not one. For a function
$f: \mathbb{R}^n \to \mathbb{R}$, the **partial derivative** with respect
to $x_i$ is the slope you get if you wiggle only $x_i$ and hold the others
fixed: $\partial f / \partial x_i$.

The **gradient** $\nabla f(x)$ is the vector of all partial derivatives:

$$\nabla f(x) = \left[\frac{\partial f}{\partial x_1}, \dots, \frac{\partial f}{\partial x_n}\right].$$

Two facts about $\nabla f$ that you must internalise:

1. $\nabla f(x)$ points in the **direction of steepest ascent** of $f$ at
   $x$. Its negative, $-\nabla f(x)$, points in the direction of steepest
   descent. This is why every optimiser in ML at heart is "step a little
   bit in the direction of $-\nabla f$".
2. At a local minimum, $\nabla f(x) = 0$. The converse is not true — a
   zero gradient also marks maxima and saddle points — but checking
   $\|\nabla f\|$ is the easiest first diagnostic when training looks
   stuck.

### Example: gradient of squared error

For a single example, the loss is $L(w) = (w \cdot x - y)^2$. With the
chain rule (next section),

$$\nabla_w L = 2(w \cdot x - y) \cdot x.$$

Read the shape: $w \cdot x - y$ is a scalar; $x$ is a vector; the product
is a vector of the same shape as $w$. That is the recipe for the gradient
of every linear model.

## The chain rule

Almost every loss in ML is a composition of simpler functions. The chain
rule tells you how to differentiate a composition: if $y = g(u)$ and
$u = f(x)$, then

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}.$$

For multivariate functions this becomes matrix multiplication of
Jacobians. The **Jacobian** of $f: \mathbb{R}^n \to \mathbb{R}^m$ is the
$m \times n$ matrix of all partial derivatives — generalising the
gradient (which is the special case $m = 1$).

The crucial ML-engineering takeaway: **backpropagation is the chain rule
applied repeatedly**. PyTorch's autograd builds a graph of operations
during the forward pass, then walks the graph backwards multiplying local
Jacobians together. You should be able to look at a model's forward pass
and trace, at least in your head, what the backward pass would compute.

### Worked example: logistic regression

Forward pass for one example: $z = w \cdot x + b$, $\hat{y} = \sigma(z)$,
$L = -[y \log \hat{y} + (1 - y) \log (1 - \hat{y})]$ (binary
cross-entropy).

Backward pass — chain rule, link by link:

- $\partial L / \partial \hat{y} = -y/\hat{y} + (1-y)/(1-\hat{y})$.
- $\partial \hat{y} / \partial z = \sigma(z)(1 - \sigma(z))$.
- $\partial z / \partial w = x$.

Multiplying through, a delightful cancellation happens:

$$\frac{\partial L}{\partial w} = (\hat{y} - y) \cdot x, \qquad \frac{\partial L}{\partial b} = \hat{y} - y.$$

That clean form — *gradient = (prediction error) × input* — is not a
coincidence. It is a deep feature of pairing cross-entropy loss with the
sigmoid; chapter 06 returns to it.

## Numerical vs. analytic gradients

Two ways to compute a gradient:

- **Analytically** — derive the formula, then implement it. Fast and
  exact. This is what library code does.
- **Numerically** — pick a small $\epsilon$ and approximate
  $\partial f / \partial x_i \approx (f(x + \epsilon e_i) - f(x - \epsilon e_i)) / (2\epsilon)$
  by perturbing one input at a time. Slow but easy to write.

The standard technique to **gradient-check** a hand-written backward pass
is to compare it against the numerical estimate on a tiny network. PyTorch
ships `torch.autograd.gradcheck` for this; you should use it any time you
implement a custom layer or loss.

## A short note on convexity (preview)

A function is **convex** if every chord between two points on its graph
lies above the graph itself. Convex functions have one minimum, and
gradient descent will find it. Squared-error loss in linear regression is
convex; cross-entropy loss in logistic regression is convex.

Neural-network losses are **non-convex** — they have many local minima
and saddle points. Gradient descent still works in practice (a fact that
took the field a decade to come to terms with), but it is no longer
guaranteed to find the global minimum. Chapter 06 picks this thread back
up.

## Automatic differentiation in one paragraph

You will never hand-compute gradients in production. Frameworks do it for
you via **automatic differentiation**: every operation in the forward
pass records itself in a computational graph; the backward pass walks the
graph in reverse, multiplying local Jacobians. This is *neither*
numerical (it is exact, modulo floating point) *nor* symbolic (it doesn't
return a formula). It is its own thing, and it is what makes modern deep
learning practical.

When you write `loss.backward()` in PyTorch you are invoking this
machinery — but you should still understand it well enough to know why a
`.detach()` in the wrong place breaks training, why `requires_grad=True`
matters, and why you sometimes have to call `optimizer.zero_grad()`.

## Summary

- The derivative is the slope of the local linear approximation; the
  gradient is the multivariable extension.
- Gradient direction = steepest ascent; negative gradient = steepest
  descent. This is the engine of training.
- The chain rule lets you differentiate compositions of functions. Done
  repeatedly, it *is* backpropagation.
- Convex losses have one minimum; deep-learning losses don't, but
  gradient descent works anyway.
- Frameworks do the differentiation; your job is to read their output.

## Where this goes next

Chapter 06 turns these tools into the gradient-descent loop you'll see
in every PyTorch training script. Before that, chapters 04 and 05 add
the probability and statistics needed to motivate where loss functions
come from in the first place.
