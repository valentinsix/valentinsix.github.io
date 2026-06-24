---
layout: page
permalink: /research/
title: Research Interests
description:
nav: true
nav_order: 2
---

I am particularly interested in the **foundations and interpretability of machine learning**. Modern learning systems can do genuinely impressive things, but we still have a surprisingly incomplete picture of what happens inside them — both while they learn and when they produce an answer. This gap is fascinating on its own, but it also leaves important practical questions open: when should we trust a model, why does it fail, and how can we make it more reliable, safe, and robust?

Rather than treating models as black boxes, I want to understand the mechanisms and building blocks that produce their behavior. Two lenses I find especially promising are **training dynamics** and **learned representations**: what does a network learn internally, how does that structure emerge over training, and why does one solution appear rather than another?

My preferred starting point is usually an empirical phenomenon that feels surprising or poorly understood. I want to characterize it carefully, identify a plausible mechanism, and then ask whether that explanation predicts something new. The aim is not only to describe what a model did, but to understand why it happened.

## Questions I Find Exciting

### How do representations take shape during training?

Representations do not always emerge gradually. Grokking [[Power et al., 2022](https://arxiv.org/abs/2201.02177)] is a striking example: a model can appear to memorize for a long time and then suddenly generalize, even after its training loss has already flattened. What changes inside the network at that moment? What controls the timing of the transition? And do similar transitions happen during large-scale pretraining, where they may be harder to see?

The lazy-to-rich transition [[Chizat et al., 2019](https://arxiv.org/abs/1812.07956); [Woodworth et al., 2020](https://arxiv.org/abs/2002.09485)] offers one useful way to think about this. Depending on choices such as initialization scale, gradient descent may stay close to a kernel-like regime or discover genuinely new features. I am interested in how this transition interacts with the structure of the data [[Damian et al., 2022](https://arxiv.org/abs/2206.01820)] and what it can tell us about feature formation in realistic models.

### When does a model memorize, and when does it generalize?

Memorization is often treated as the opposite of generalization, but the picture is more complicated. For example, memorizing rare examples may sometimes be necessary for good generalization [[Feldman, 2020](https://arxiv.org/abs/1911.05451)]. I want to understand which properties of the data, model, and training trajectory push learning toward one regime or the other, and how that choice is reflected in the model's internal representations.

This question also matters beyond test accuracy. It may help explain when capabilities emerge, how models learn undesirable behaviors, and which examples have a lasting influence on what a model becomes.

### Where does in-context learning come from?

In-context learning is a remarkable capability: a language model can adapt its behavior from examples in a prompt without any gradient update. We have several promising accounts — including induction heads [[Olsson et al., 2022](https://arxiv.org/abs/2209.11895)], Bayesian inference [[Xie et al., 2022](https://arxiv.org/abs/2111.02080)], and implicit gradient descent [[Akyürek et al., 2022](https://arxiv.org/abs/2211.15661)] — but it is still difficult to predict when and why this ability will emerge.

I am especially interested in what pretraining must install for in-context learning to become possible. Its emergence appears to depend on statistical properties of the training distribution [[Chan et al., 2022](https://arxiv.org/abs/2205.05055)], which makes it a natural meeting point between training dynamics, data structure, and mechanism-level interpretability.

### Can we predict systematic failures before observing them?

The capability profile of modern language models is often strange: they can perform well on difficult benchmarks and still fail on simple compositional variations. Cataloguing these failures is useful, but I am more interested in explanations that predict them. Can a model's architecture, training dynamics, or internal representations tell us in advance which tasks will be brittle and why?

For me, this is an important test of a mechanistic explanation. A good account should do more than fit a story to a known failure; it should make specific claims about failures we have not yet measured.

### Which mechanisms are universal?

Interpretability becomes much more useful if the mechanisms found in one model also appear in other architectures and at other scales. There is empirical evidence that learned representations become more similar as models improve [[Huh et al., 2024](https://arxiv.org/abs/2405.07987)], but we do not yet have a satisfying explanation for when this convergence should occur.

I would like to understand whether recurring mechanisms are accidents of particular architectures or natural solutions to common learning problems. The answer determines how much we can learn from small, controlled models about much larger systems.

## What Connects These Threads

All of these questions point back to the learning process itself: how training choices, data, and architecture shape internal structure, and how that structure becomes behavior. I am most excited by explanations that make model behavior feel less mysterious — not because they provide a convincing story after the fact, but because they are precise enough to be tested and useful enough to predict something new.
