# quiz-01 — ML Foundations: Math, Statistics, and Problem Framing

> **Format:** 12 multiple-choice questions, one correct answer per question.
> **Covers:** chapters 01–08 of this module.
> **How to use:** read each question, pick one answer, then check the key at
> the bottom. The explanation after each answer names the chapter the point
> comes from — go re-read that section if you missed the question.

---

## Questions

### 1. (Chapter 01) Why does this module open with a maths refresher?

A. Because a working ML engineer is expected to prove convergence results
   on the job.
B. Because every loud failure in a working ML team eventually traces back
   to a miscommunication with the underlying maths.
C. Because interview panels for ML engineering roles are
   maths-heavy and the curriculum optimises for that.
D. Because the libraries you will use (NumPy, PyTorch) require you to
   implement every operation from scratch.

### 2. (Chapter 02) `A` has shape `(32, 128)` and `B` has shape `(128, 10)`. What is the shape of `A @ B`?

A. `(32, 128, 10)`
B. `(32, 10)`
C. `(10, 32)`
D. `A @ B` is not defined because the shapes do not match.

### 3. (Chapter 02) In a batch `X` of shape `(m, n)`, which statement matches the convention described in the chapter?

A. `m` is the number of features and `n` is the number of examples.
B. `m` is the number of examples and `n` is the number of features.
C. `m` and `n` are interchangeable; convention does not matter.
D. `m` is the embedding dimension and `n` is the batch size.

### 4. (Chapter 03) For `L(w) = (w · x − y)^2` with a single example `(x, y)`, what is `∇_w L`?

A. `(w · x − y)^2 · x`
B. `2 (w · x − y) · x`
C. `2 (w · x − y)`
D. `x − y`

### 5. (Chapter 03) The gradient `∇f(x)` is zero. Which of the following is **not** necessarily true?

A. `x` is a stationary point of `f`.
B. `x` could be a saddle point of `f`.
C. `x` could be a local maximum of `f`.
D. `x` is a local minimum of `f`.

### 6. (Chapter 04) The probability density function `p(x)` of a continuous random variable evaluated at a point `x = 2.0` returns the number `0.4`. Which interpretation is correct?

A. There is a 40% chance that the random variable equals `2.0`.
B. `0.4` is a probability per unit length; the probability of any single
   exact point is zero, and `P(a ≤ X ≤ b) = ∫_a^b p(x) dx`.
C. The random variable cannot exceed `0.4`.
D. `p(x)` is only defined for discrete random variables.

### 7. (Chapter 04) A naive Bayes classifier assumes that the features are:

A. Independent of each other and of the label.
B. Identically distributed.
C. Conditionally independent given the label.
D. Linearly separable.

### 8. (Chapter 05) In MAP estimation, a Laplace prior on the weights produces which regulariser?

A. `L2` (ridge) regularisation.
B. `L1` (lasso) regularisation.
C. Early stopping.
D. Dropout.

### 9. (Chapter 05) Why is squared error the "natural" loss for regression under the probabilistic view presented in the chapter?

A. It is the only loss whose gradient can be computed in closed form.
B. It is the negative log-likelihood of the labels under an assumed
   Gaussian noise model on the residual.
C. It is required by the universal approximation theorem.
D. It is the only convex loss function.

### 10. (Chapter 06) Training with one feature in `[0, 1]` and another in `[0, 10^6]` tends to make plain gradient descent "zig-zag" and crawl. Which fix does the chapter recommend as the cheapest first intervention?

A. Switch to L-BFGS.
B. Switch to Adam with a very small learning rate.
C. Standardise the features to zero mean and unit variance.
D. Add L1 regularisation so the large-scale feature is zeroed out.

### 11. (Chapter 07) According to the chapter, which situation is the one in which reinforcement learning is genuinely the right framing (rather than supervised learning)?

A. There are no labels at all, so RL is used as a label-free fallback.
B. The data your model sees in deployment depends on the actions the
   model takes.
C. The dataset is very small, so RL is used because it is sample-efficient.
D. The target variable is categorical with more than two classes.

### 12. (Chapter 08) A fraud-detection model is trained on data where only 1% of transactions are fraud. The team reports 99% accuracy and ships it. Which chapter-08 critique applies most directly?

A. Accuracy is hiding the class imbalance; "predict `not fraud`" scores
   99% without catching any fraud. The team should look at
   precision / recall / PR-AUC and tune the decision threshold.
B. The team should have used regression instead of classification so that
   the output is a continuous score.
C. The team should have used a Laplace prior to produce `L1` sparsity.
D. The team should have reframed the problem as reinforcement learning
   because the actions (block / allow) are being taken in production.

---

## Answer key

<details>
<summary>Click to expand (or scroll past if you want to self-check first).</summary>

1. **B.** Chapter 01 motivates the entire module with the claim that
   "every loud failure in a working ML team eventually traces back to a
   miscommunication with the maths" — the maths is a tool for diagnosis,
   not a hoop to jump through.

2. **B.** Chapter 02 ("Matrix multiplication"): `(m, k) @ (k, n) → (m, n)`.
   The inner dimensions must match and are consumed; the outer dimensions
   survive. So `(32, 128) @ (128, 10) → (32, 10)`.

3. **B.** Chapter 02 ("Matrices"): the convention is examples down the
   rows and features across the columns, so a batch of `m` examples with
   `n` features is a matrix in `ℝ^{m × n}`.

4. **B.** Chapter 03 ("Example: gradient of squared error"): applying the
   chain rule to `L = (w · x − y)^2` gives `∇_w L = 2 (w · x − y) · x` —
   a vector-shaped like `w`.

5. **D.** Chapter 03 ("Partial derivatives and the gradient"): a zero
   gradient marks a stationary point, but the converse of "local
   minimum ⇒ zero gradient" is not true — zero gradient also happens at
   maxima and saddle points.

6. **B.** Chapter 04 ("Random variables"): a density is a probability
   *per unit length*, not a probability. Only its integral over an
   interval is a probability.

7. **C.** Chapter 04 ("Independence vs. i.i.d. vs. conditional
   independence"): the chapter calls this out explicitly — naive Bayes
   assumes features are conditionally independent *given the label*.

8. **B.** Chapter 05 ("Maximum a posteriori (MAP) estimation"): L2
   regularisation corresponds to a Gaussian prior; L1 regularisation
   corresponds to a Laplace prior.

9. **B.** Chapter 05 ("MLE recovers familiar losses"): under
   `y_i = w · x_i + ε_i` with `ε_i ~ 𝒩(0, σ^2)`, the NLL of the labels
   is the sum of squared errors up to a positive constant.

10. **C.** Chapter 06 ("Gradient descent", knob 2 — feature scale):
    standardising features (zero mean, unit variance) is called out as
    "the single cheapest fix and one of the highest-leverage habits in
    classical ML."

11. **B.** Chapter 07 ("When RL fits" / "A useful heuristic"): RL is the
    right framing specifically when the actions your model takes
    meaningfully change the future inputs it sees — otherwise supervised
    learning is faster and more reliable.

12. **A.** Chapter 08 ("Class imbalance"): accuracy on an imbalanced
    dataset is the biggest classification footgun — predicting the
    majority class scores high without doing anything useful. The
    chapter's recommended responses are a better metric
    (precision / recall / PR-AUC) and a tuned decision threshold.

</details>
