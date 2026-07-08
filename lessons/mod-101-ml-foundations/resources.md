# Resources for mod-101 — ML Foundations: Math, Statistics, and Problem Framing

A curated short list. The goal is depth-on-demand for the specific
chapter you got stuck on, not a comprehensive bibliography. Primary
documentation and canonical textbooks are listed first; lecture series
and writeups follow.

## Primary documentation

- **NumPy user guide — broadcasting.**
  <https://numpy.org/doc/stable/user/basics.broadcasting.html>
  Read once carefully — it is the source of half the shape bugs you
  will encounter.
- **NumPy `linalg` reference.**
  <https://numpy.org/doc/stable/reference/routines.linalg.html>
- **PyTorch — broadcasting semantics.**
  <https://pytorch.org/docs/stable/notes/broadcasting.html>
- **PyTorch — autograd mechanics.**
  <https://pytorch.org/docs/stable/notes/autograd.html>
- **PyTorch — `torch.autograd.gradcheck`.**
  <https://pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html>
- **SciPy — statistical functions catalogue.**
  <https://docs.scipy.org/doc/scipy/reference/stats.html>
  The canonical reference for the distributions covered in chapter 04.
- **scikit-learn — choosing the right estimator.**
  <https://scikit-learn.org/stable/tutorial/machine_learning_map.html>
  A diagram you'll return to in `mod-102`/`mod-104`; useful framing
  reference here.

## Textbooks (canonical)

- **Strang, Gilbert. *Introduction to Linear Algebra* (6th ed.,
  Wellesley-Cambridge Press).** The standard reference for chapter 02.
  Companion lectures are free on MIT OCW (course 18.06).
- **Goodfellow, Bengio, and Courville. *Deep Learning* (MIT Press,
  2016).** Free HTML at <https://www.deeplearningbook.org>.
  Part I (chapters 2–5) is a self-contained refresher on linear
  algebra, probability, numerical computation, and the basics of
  machine learning — the closest single source to this module's
  content at the right altitude.
- **Bishop, Christopher M. *Pattern Recognition and Machine
  Learning* (Springer, 2006).** Chapters 1–2 cover probability and
  decision theory; chapter 8 covers graphical models and the
  conjugate-prior view of MAP that chapter 05 leans on.
- **Murphy, Kevin P. *Probabilistic Machine Learning: An
  Introduction* (MIT Press, 2022).** Free PDFs at
  <https://probml.github.io/pml-book/book1.html>.
  A modern, code-backed treatment of all of chapters 03–06 of this
  module.
- **Hastie, Tibshirani, and Friedman. *The Elements of Statistical
  Learning* (2nd ed., Springer, 2009).** Free PDF at
  <https://hastie.su.domains/ElemStatLearn/>. Sections 2–4 are
  excellent on regression, classification, and the bias-variance
  trade-off.
- **James, Witten, Hastie, and Tibshirani. *An Introduction to
  Statistical Learning* (Springer; Python edition 2023).** Free at
  <https://www.statlearning.com/>. A friendlier on-ramp than ESL;
  read alongside chapters 04–05 if statistical estimation feels
  heavy.
- **Huyen, Chip. *Designing Machine Learning Systems* (O'Reilly,
  2022).** Chapter 2 ("Introduction to Machine Learning Systems
  Design") is the production-engineering counterpart to chapter 07
  of this module.
- **Burkov, Andriy. *Machine Learning Engineering* (True Positive
  Inc., 2020).** Chapters on problem definition and goal-setting
  pair naturally with chapter 07.

## Lecture series

- **MIT 18.06 — *Linear Algebra* (Gilbert Strang).** Free lectures on
  MIT OpenCourseWare. The "matrix multiplication is many things at
  once" framing in lecture 3 is the cleanest treatment I know.
- **3Blue1Brown — *Essence of Linear Algebra* and *Essence of
  Calculus*.** YouTube series. Geometric intuition that pairs well
  with chapters 02 and 03.
- **Andrew Ng — *Machine Learning Specialization* (Coursera /
  DeepLearning.AI, current revision).** The chapters on linear /
  logistic regression and gradient descent are the gentlest possible
  on-ramp to chapters 03 and 06.

## Articles and references for the framing chapters

- **Google — *Rules of Machine Learning: Best Practices for ML
  Engineering*.**
  <https://developers.google.com/machine-learning/guides/rules-of-ml>
  Practical, opinionated, free. Many of the chapter-07 instincts
  trace back to this document; cite it.
- **scikit-learn — *Underfitting vs. Overfitting* tutorial.**
  <https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html>
- **Goyal et al. — *Accurate, Large Minibatch SGD: Training ImageNet
  in 1 Hour*.** <https://arxiv.org/abs/1706.02677>
  Source of the "linear scaling rule" cited in chapter 06.
- **Kingma & Ba — *Adam: A Method for Stochastic Optimization*.**
  <https://arxiv.org/abs/1412.6980>
- **Loshchilov & Hutter — *Decoupled Weight Decay Regularization*
  (AdamW).** <https://arxiv.org/abs/1711.05101>

## Cheat-sheets and quick references

- **The Matrix Cookbook** (Petersen & Pedersen).
  <https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf>
  Reference for matrix derivative identities you'll hit in chapters
  05–06.
- **A few useful things to know about ML** (Pedro Domingos, CACM
  2012). <https://homes.cs.washington.edu/~pedrod/papers/cacm12.pdf>
  An older but still excellent short article on the practitioner's
  view of ML.

<!-- needs-research: confirm latest editions and free-availability URLs for the textbooks listed above on the next autonomous research cycle; replace any link that has moved or 404'd. -->
