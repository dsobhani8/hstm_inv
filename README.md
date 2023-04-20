# Invariant Heterogeneous Supervised Topic Models

This repository leverages code from:
  [1] Heterogeneous Supervised Topic Models, by Dhanya Sridhar, Hal Daumé III, and David Blei.
  
Please visit to see the [original paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00487/111727/Heterogeneous-Supervised-Topic-Models) and for the [original code](https://github.com/dsridhar91/hstm)

In this repository, we introduce two new datasets with spurious correlations induced and added the anti-causal regularizer from Counterfactual Invariance to Spurious Correlations (Veitch et al., 2021).

**To run the experiment from the write-up in colab please run the following code:**

```
!git clone https://github.com/dsobhani8/hstm_inv
```

```
!pip install dill nltk numpy pandas scikit-learn scipy tensorflow tensorflow-hub tensorflow-probability tokenizers torch torchvision transformers
```

```
%cd /content/hstm_inv/src
```

```
!python -m experiment.run_experiment --procfile=/content/hstm_inv/dat/proc/comp_spurious_train_spurious_test.npz --data=custom --train_size=5000 --num_topics=20 --batch_size=32 --train_test_mode=True --do_pretraining_stage=True --is_MMD=True --MMD_pen_coeff=2
 ```
 
**The two new datasets are:**

(1) comp_spurious_train_spurious_test.npz

(2) comp_spurious_train_non_spurious_test.npz


**The two new flags are:**

(1) is_MMD (default is False)

(2) MMD_pen_coeff (default is 2)


It is important to keep data=custom and train_size=5000

**References:**

[1] Heterogeneous Supervised Topic Models, by Dhanya Sridhar, Hal Daumé III, and David Blei.

[2] Counterfactual Invariance to Spurious Correlations: Why and How to Pass Stress Tests, by Victor Veitch, Alexander D’Amour, Steve Yadlowsky, and Jacob Eisenstein

