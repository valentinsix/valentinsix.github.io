---
layout: page
permalink: /research/
title: Research Interests
description:
nav: true
nav_order: 2
---

I'm interested in the **foundations and interpretability of machine learning**. What pulls me in is a simple observation: we have systems that do genuinely impressive things, and we still don't really know what's happening inside them — both while they learn and when they produce an answer.

I don't have a single tight research question yet. I'm still exploring, reading broadly, and following threads that feel worth following. Right now I'm mostly finding problems and patterns — noticing what seems surprising, and figuring out which questions I actually care about.

The two lenses I keep coming back to are **training dynamics** and **learned representations**: what does a network actually learn internally, how does that structure emerge over training, and why does one solution appear rather than another?

## Questions I Find Exciting

### How do representations take shape during training?

Representations don't always emerge gradually. Grokking [[Power et al., 2022](https://arxiv.org/abs/2201.02177)] shows that a model can appear to memorize for a long time and then suddenly generalize, even after training loss has already flattened. What changes inside the network at that moment? And do similar transitions happen during large-scale pretraining, where they may be harder to see?

**Developmental interpretability** is a direction I find very intuitive: the idea is to study not just what a trained model does, but how it got there, and what structures appear and disappear during training. It views interpretability as a process, not a post-hoc analysis.

<!-- The lazy-to-rich transition [[Chizat et al., 2019](https://arxiv.org/abs/1812.07956); [Woodworth et al., 2020](https://arxiv.org/abs/2002.09485)] is another useful lens. Depending on initialization scale, gradient descent may stay close to a kernel-like regime or discover genuinely new features. I'm curious how this interacts with data structure [[Damian et al., 2022](https://arxiv.org/abs/2206.01820)] and what it tells us about how features actually form in real models. -->

### When does a model memorize, and when does it generalize?

The standard story is that memorization and generalization are opposites — a model either memorizes its training data or learns something transferable. But that's not quite right: memorizing rare examples can actually be necessary for good generalization [[Feldman, 2020](https://dl.acm.org/doi/pdf/10.1145/3357713.3384290)]. **Singular learning theory** studies the geometry of the loss landscape near singularities to predict which solutions gradient descent actually finds, and why some generalize better than others.

This question also matters beyond test accuracy ; it may help explain when capabilities emerge and which examples have a lasting influence on what a model becomes.

### Where does in-context learning come from?

In-context learning is genuinely strange: a language model can adapt from examples in a prompt without any gradient update. There are several good explanations: induction heads [[Olsson et al., 2022](https://arxiv.org/abs/2209.11895)], Bayesian inference [[Xie et al., 2022](https://arxiv.org/abs/2111.02080)], preconditionned gradient descent [[Ahn et al., 2024](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8ed3d610ea4b68e7afb30ea7d01422c6-Abstract-Conference.html)] — but none of them tells you when or why this ability will emerge.

I'm especially interested in what pretraining must spark for in-context learning to become possible. Its emergence seems to depend on statistical properties of the training distribution [[Chan et al., 2022](https://arxiv.org/abs/2205.05055)], which ties together training dynamics, data structure, and interpretability in one place.

### Can we predict systematic failures before observing them?

Language models have a strange failure mode: ace a hard benchmark, then break on a simple rephrasing. You can keep cataloguing these cases, but at some point you want an explanation that gets ahead of them. Can a model's architecture, training dynamics, or internal representations tell you in advance which tasks will be brittle, or which behaviours are emerging in the model?

This is an interesting test to apply to any mechanistic account: not whether it explains a known failure, but whether it predicts ones we haven't measured yet.

### Which mechanisms are universal?

A circuit found in one model is an interesting curiosity. A circuit found across architectures and scales is much more important. Models trained independently tend to converge on similar representations as they improve [[Huh et al., 2024](https://arxiv.org/abs/2405.07987)], but we don't have a good explanation for when or why this happens.

Whether recurring mechanisms are accidents of particular architectures or something more fundamental matters a lot, for instance because it determines how much you can learn from small models about much larger ones.

---

Overall, I like to find surprising phenomena in ML and go back to the theory to understand what's going on.
