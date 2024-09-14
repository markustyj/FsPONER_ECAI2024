---
language:
- en
tags:
- security-domain-adaptation
- vilocify
license: Siemens Inner Source License v1.3
datasets:
- dataset_0: vilocify
metrics:
- perplexity

model-index:
- name: security-vilocify
  results:
    - task:
          type: it-sec-mlm
          name: IT Security Language Model
      dataset:
          type: vilocify
          name: Vilocify
      metrics:
        - type: perplexity
          value: 2.4

---

# IT Security Language Model

## Model description

You can embed local or remote images using `![](...)`

## Intended uses & limitations

#### How to use

```python
# You can include sample code which will be formatted
```

#### Limitations and bias

Provide examples of latent issues and potential remediations.

## Training data

Describe the data you used to train the model.
If you initialized it with pre-trained weights, add a link to the pre-trained model card or repository with description of the pre-training data.

## Training procedure

Preprocessing, hardware used, hyperparameters...

## Eval results

### BibTeX entry and citation info

```bibtex
@inproceedings{...,
  year={2020}
}
```
