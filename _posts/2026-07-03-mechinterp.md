---
layout: post
title: A Gentle Introduction to Mechanistic Interpretability
date: 2026-07-03
description: Laying down the basics and justifying the existence of the field.
published: true
tags:
categories:
---

I sometimes find myself explaining mechanistic interpretability to people from scratch, so I figured I'd write the explanation down once, mostly to avoid vague explanations. If you've trained a few neural networks, you already know the uncomfortable truth about deep learning: we're very good at getting these models to work, and not very good at explaining _why_ they work. You can train a transformer to write code, translate languages, or beat you at chess, but ask "how exactly is it doing that?" and the honest answer, for most of deep learning's history, has been a shrug and a matrix of a billion floating point numbers.

**Mechanistic interpretability** (often shortened to "mech interp" or "MI") is a research field trying to change that. The goal isn't just to say "this input leads to that output." It's to reverse-engineer the algorithm a trained network has learned, in enough detail that you could, in principle, rewrite it as normal code. This post is a friendly introduction with enough background to understand the vocabulary, the main ideas, and why people find this exciting (and difficult).

I'll assume you know the basics of neural networks and a bit of linear algebra, but nothing about interpretability specifically.

---

## Why bother looking inside at all?

There are a few different motivations that draw people into this field, and it's worth separating them because they lead to different research styles.

**Safety:** as models get more capable, we'd like some guarantee that they're doing what we think they're doing, for the reasons we think they're doing it. A model that's honest by coincidence looks identical from the outside to a model that's honest because it has learned to appear that way when watched. If we can't look inside, we can't tell these apart. Interpretability is one of the few tools that could, in principle, distinguish "actually aligned" from "successfully faking it."

**Scientific curiosity:** separately from safety, this is just a fascinating scientific question. Deep learning produces systems that do impressive things, and we mostly don't know how. That's an intellectually strange position to be in as a field: it would be like chemistry with no periodic table, just "put things together, see what happens." Mechanistic interpretability is an attempt to build a periodic table for what's going on inside trained networks.

**Improving the models:** more practically, understanding internal mechanisms helps you fix models. If you know _why_ a model hallucinates, or _why_ it fails on a certain class of inputs, you can fix the actual problem instead of patching the symptom with more training data.

All three motivations point to the same technical project: open up the black box and find the algorithm inside. The scientific angle is what pulled me into the field, but the safety aspect is a concrete and industry-related one, which makes the problem interesting AND important.

---

## The core idea: networks as programs

The central move in mechanistic interpretability is a shift in mindset. Instead of thinking of a neural network as _a function that was fit to data_, you think of it as _a program that was compiled by gradient descent_. Somewhere inside those weights is a set of subroutines: computational structures that implement specific behaviors. And our job is of course to find them.

But this might sound like a strange thing to hope for: why would gradient descent even produce anything as clean as a "subroutine"? The empirical answer, from several years of work by researchers (Anthropic, Redwood Research, and various other academic labs) is: it does, and more often than you'd really expect. Networks seem to learn genuinely modular, reusable pieces of computation, and when you find them, they sometimes make intuitive sense.

The two central objects in this program-reconstruction project are **features** and **circuits**.

---

## Features, a.k.a. the "variables" of the program

A **feature** is (informally) a property of the input that the network represents internally: something that's "on" for some inputs and "off" (or scaled) for others. Classic examples from vision models: a "curve detector" neuron, a "dog ear" detector, a "car wheel" detector. In language models: a feature that fires on legal text, a feature that tracks whether we're inside quotation marks, a feature that fires on grammatical subjects...

If neurons neatly corresponded to features (one neuron = one concept), mechanistic interpretability would be almost too easy: you'd just look at what makes each neuron fire, label it, and move on. Early interpretability work on vision networks found many neurons like this. But as people looked more carefully, especially in language models, this clean picture broke down.

---

## Superposition: too many concepts, not enough neurons

If you look at an individual neuron in a language model, it often seems to respond to a weird, unrelated basket of things: say, it fires on French text, on mentions of the number seven, and on requests for recipes. This is called **polysemanticity**: one neuron = many unrelated meanings.

Why would a network do this? The leading explanation is a phenomenon called **superposition**. The idea, roughly:

- The world has _far more_ meaningful features in it than any network has neurons available to represent them.
- Real-world features are usually **sparse**: most concepts are absent most of the time (a given sentence usually isn't about French, the number seven, and recipes all at once; actually most of the time none of these are active).
- Because most features are sparse, a network can get away with squeezing _many more features than it has dimensions_ into its representation space, as long as those features rarely need to be active simultaneously. When they do overlap, you get a bit of interference (noise), but if it's rare enough, it's a price worth paying compared to only representing as many concepts as you have neurons.

Analogy time: imagine coat-check system where you're allowed to hang multiple coats on one hook, because you know it's unlikely that all the coats sharing a hook get claimed on the same day. Sometimes two people show up for the same hook and there's a mix-up. But you can serve far more customers this way than if you insisted on one coat per hook.

To be slightly more formal, lhe **linear representation hypothesis** (a working assumption behind most of this research) says that a layer's activation vector $x \in \mathbb{R}^d$ can be written as a sparse combination of feature directions:

$$
x = \sum_{i=1}^{n} f_i \, v_i
$$

where each $v_i \in \mathbb{R}^d$ is a fixed direction representing "feature $i$," each $f_i \geq 0$ tells you how strongly that feature is active for this particular input, and only a small number of the $f_i$ are non-zero at once (hence the sparsity).

A key detail is that the number of features $n$ can be (and often is) **much larger than the number of dimensions $d$** available to represent them ($n \gg d$). That's only possible geometrically because the $v_i$ are not orthogonal: they overlap a bit, so activating one feature leaks a small amount into the directions of others. As long as features are rarely on at the same time, that leakage stays small and the network can still mostly recover which features were active.

Superposition means that individual neurons are generally **not** the right unit of analysis. Real features live along _directions_ in the high-dimensional activation space, and many of them don't line up neatly with any single neuron's axis. This is one of the main reasons mechanistic interpretability is hard: the thing you actually want to look at (a feature) doesn't correspond to something you can directly read off (a neuron).

### Unmixing features with Sparse Autoencoders

If features are directions in activation space rather than individual neurons, we need a way to find those directions. The current favorite tool is the **sparse autoencoder (SAE)**.

The basic idea is to train a small auxiliary network that takes a layer's activations, projects them into a much _higher-dimensional_ space (more "slots" than the original number of neurons), and reconstructs the original activations, while forcing most of the high-dimensional slots to be zero (inactive) for any given input. If this works, each of those slots tends to correspond to something close to a genuine, monosemantic feature (a single interpretable concept) rather than a polysemantic neuron.

Concretely, an SAE learns an encoder and decoder:

$$
z = \text{ReLU}(W_{\text{enc}} \, x + b_{\text{enc}}), \qquad \hat{x} = W_{\text{dec}} \, z + b_{\text{dec}}
$$

where $z \in \mathbb{R}^m$ with $m \gg d$ (many more slots than the original layer had neurons), and it's trained to minimize a loss that trades off reconstruction accuracy against sparsity of $z$:

$$
\mathcal{L} = \underbrace{\|x - \hat{x}\|_2^2}_{\text{reconstruction}} + \lambda \underbrace{\|z\|_1}_{\text{sparsity penalty}}
$$

The $\ell_1$ penalty pushes most entries of $z$ to exactly zero for any given input, which is what encourages each surviving non-zero entry to correspond to a single, clean, interpretable feature rather than a blend of several. This has been one of a very productive development in the field, as it gives you a somewhat automated way of pulling human-interpretable concepts out of superposed representations, at least approximately.

{% include figure.liquid path="assets/img/mechinterp/sae-diagram.png" class="img-fluid rounded z-depth-1 mx-auto d-block diagram-on-light" max-width="480px" zoomable=true caption='A sparse autoencoder reconstructing a target network&#39;s activations through a wider, sparsely-active hidden layer, in the hope that each active unit lines up with a single genuine feature. <a href="https://www.alignmentforum.org/posts/tLCBJn3NcSNzi5xng/deep-sparse-autoencoders-yield-interpretable-features-too" target="_blank">Source</a>.' %}

---

## Circuits: features wired together

Finding individual features is cool, but not nearly enough to do something useful with it. A program isn't just a list of variables, they also need to be _connected by computation_. In mech interp, the analogous idea is a **circuit**: a subgraph of the network (a specific set of features, or neurons/attention heads, and the connections between them) that together implement some identifiable piece of behavior.

For example: a circuit might be the specific combination of attention heads and MLP neurons that lets a language model figure out that in the sentence _"When Mary and John went to the store, John gave a drink to \_\_\_"_, the blank should be "Mary" rather than "John." That's a real, well-studied example (the "Indirect Object Identification" circuit, more on it below). Researchers have traced out, head by head, how the model tracks who has already been mentioned, who the subject is, and who the object should be.

Once you have both features and circuits, you have the two ingredients of the reverse-engineering project: _what_ the network represents, and _how_ those representations get combined to produce outputs.

---

## The toolbox

There are standard techniques that come up repeatedly. It's maybe worth knowing their names, since they show up in most papers in the domain.

**Probing:** train a small, simple classifier (like linear regression) on a model's internal activations to see if some property of interest (e.g., "is this token inside a negation?") is linearly recoverable from them. If a simple probe can extract it easily, the model likely represents that property in a fairly clean, accessible way. Probing tells you _that_ information is present, but not that the model actually _uses_ it. For that, you need causal methods.

**Ablation:** zero out (or otherwise remove) a component (a neuron, an attention head, a whole layer) and see what breaks. If deleting a specific attention head destroys the model's ability to do subject-verb agreement but doesn't touch anything else, that's evidence the head is doing something specific and identifiable.

**Activation patching / causal tracing:** you run the model on two versions of an input that differ in one key way: a "clean" run and a "corrupted" run (e.g., "The Eiffel Tower is in Paris" vs. "The Eiffel Tower is in Rome"). Then, for some component $h$ (a specific neuron, head, or layer) you copy its activation from the clean run into the corrupted run, and measure how much this shifts the output metric $M$ (e.g., the model's probability on the correct answer) back toward the clean result:

$$
\text{Effect}(h) = M\big(\text{corrupted}_{\,h \leftarrow \text{clean}}\big) - M(\text{corrupted})
$$

A large effect means that component is causally responsible for carrying the relevant information. Doing this systematically across every component gives you a map of _where_ in the network a given piece of information lives and gets used. That's how circuits like IOI (below) get traced out.

**Attribution methods:** techniques (often gradient-based) that estimate how much each internal component contributed to the final output, without needing to run the full model twice per component like ablation does. Faster, but generally a rougher approximation than direct ablation or patching.

**Sparse autoencoders**, as covered above, for unmixing superposed features into interpretable directions.

More generally, the field has moved from "purely observational" methods (probing: what's correlated with what) toward "causal" methods (patching, ablation: what actually matters for the output). Correlation-only findings can be misleading about what the model is actually doing versus what's just incidentally present. A personal comment on the current state of MI: it seems to me that most efforts in the field converge on analzying the model AFTER it has been trained. But I haven't found a lot of work focusing on _how the model in gradually learning_, and how we arrive to the final model. [Developmental Interpretability](https://valentinsix.github.io/blog/2026/slt/) is a young field that seems to focus on this gap. Or maybe this kind of work is mostly about training dynamics, and not so much "proper interpretability". If you have an opinion about this, please reach out!

---

## A few concrete case studies

Abstract descriptions only get you so far, and it is actually one of the strong criticism the filed is facing. But here are some of the field's most-cited concrete results:

**Induction heads:** it's one of the earliest and cleanest findings. Many transformers develop a specific type of attention head, called an induction head, that implements a simple pattern-completion rule. If token $A$ was followed by token $B$ earlier in the context, and $A$ shows up again, the head boosts the prediction for $B$:

$$
\text{if } \ldots, A, B, \ldots, A, \_\_ \quad \Rightarrow \quad P(\text{next} = B) \uparrow
$$

This is a big part of how models do in-context learning: copying and completing patterns they've seen earlier in their own context window. Induction heads also tend to emerge somewhat suddenly during training, and that emergence tracks with a jump in the model's in-context learning ability, a nice concrete link between "a specific mechanism appearing" and "a capability appearing."

**The Indirect Object Identification (IOI) circuit:** given a sentence like _"When Mary and John went to the store, John gave a drink to \_\_\_,"_ GPT-2 small reliably predicts "Mary." Researchers traced out the full circuit responsible: specific attention heads that identify duplicate names, other heads that figure out who the subject of the sentence is, and heads that move the correct name to the output. This was one of the first times a non-trivial language model behavior was mapped out almost completely, head by head.

**Modular addition and grokking:** A small transformer trained to compute addition modulo some prime number will, after a long plateau, suddenly "grok" the task: it goes from from memorizing the training examples to generalizing perfectly. When researchers looked inside a grokked network, they found it had learned to represent each number $a$ as a point on a circle, roughly as $(\cos(\omega a), \sin(\omega a))$ for some learned frequency $\omega$. Addition then falls out almost for free from the angle-addition identity:

$$
\cos(\omega(a+b)) = \cos(\omega a)\cos(\omega b) - \sin(\omega a)\sin(\omega b)
$$

The network's internal multiplications and additions turn out to implement essentially this trigonometric identity, which is why the result generalizes perfectly instead of just memorizing training examples: it's a genuine, compact arithmetic algorithm.

{% include figure.liquid path="assets/img/mechinterp/grokking-1-layer-transformer.webp" class="img-fluid rounded z-depth-1 diagram-on-light" zoomable=true caption='The grokked one-layer transformer&#39;s internal circuit for modular addition: numbers get embedded onto a circle, combined via trig identities, and read off into logits. <a href="https://www.neelnanda.io/grokking-paper" target="_blank">Source</a>.' %}

These examples matter beyond their specific content: they're existence proofs. They show that at least in some cases, real trained networks _do_ implement clean, human-legible algorithms. And that's the basic bet the whole field is making.

---

## Why this is hard

It's worth being honest about the obstacles, since they explain why "understand GPT-5.5 completely" isn't around the corner.

**Superposition, again:** as mentioned earlier, most interesting representations aren't cleanly aligned with individual neurons, so almost all analysis needs an "unmixing" step first, and that step is itself imperfect and an active research problem.

**Scale:** the case studies above are mostly on small models (GPT-2 scale or smaller) and narrow behaviors (a specific arithmetic task, a specific sentence pattern). Frontier models have orders of magnitude more parameters and are used for wildly more diverse tasks. Circuit-level understanding doesn't obviously scale: we don't yet have a clear plan for going from "we understand this one circuit in a small model" to "we understand a large fraction of a frontier model's behavior." It's also easiest to study things that are easy to study, which are often well-defined, narrow, static behaviors on small models. There's a real risk of the field over-indexing on convenient examples rather than the messy, high-stakes behaviors we most want to understand (deception, situational awareness, goal-directed planning...).

**No ground truth:** unlike, say, biology, where you can eventually verify a hypothesis about a mechanism by direct experiment, it's often unclear how to be fully certain an interpretability explanation is _correct_ rather than just _plausible-sounding_. This is honestly the part that frustrates me most. A lot of SAE features get called "interpretable" because a researcher scanned the examples that activate it and the label felt right, not because anyone checked that the feature actually does what the label claims. That's manual, eyeball-it, and it's the kind of manual work that doesn't obviously get more reliable just because you do more of it. Maybe I'm being too harsh here, plenty of people in the field take this seriously and are actively trying to build better validation methods, but until there's a real way to check a feature's claimed meaning against something other than "yeah, that tracks," I stay somewhat skeptical of how much SAEs have actually solved.

---

## Where the field is headed

A few current directions are worth knowing about. One is **scaling sparse autoencoders** to frontier-size models, to see whether the "unmixing into clean features" approach holds up outside of toy settings. Another is **automating interpretability**: using language models themselves to generate and test hypotheses about what other models' components are doing, since manual circuit-tracing doesn't scale to human labor alone. A third, newer offshoot is **developmental interpretability** (mentionned ealier ; see linked post), which looks at _how_ circuits and features emerge over the course of training, rather than only analyzing a finished model. And there's growing interest in **interpretability for alignment evaluation**: using internal probes as a kind of "lie detector," to try to catch cases where a model's stated reasoning doesn't match what it's actually doing internally.

---

## Wrapping up and how to start

A few concrete starting points: Anthropic's **Transformer Circuits** publications are the closest thing the field has to a canonical reading list, starting with the original circuits work on vision models and moving through induction heads, superposition, and sparse autoencoders. **TransformerLens** is an open-source library built specifically for this kind of research: it makes it easy to grab internal activations, run activation patching experiments, and poke around small models like GPT-2. Replicating a small, well-documented result yourself is a much better way to build intuition than reading about it. 

Mechanistic interpretability is, at its core, a huge bet: that the strange, huge-dimensional, seemingly opaque computations inside a trained neural network aren't actually an unstructured mess, but a genuine algorithm, messy in places, but ultimately built out of identifiable and reusable parts. The field has real, concrete wins to point to, and honest obstacles still in front of it. I don't think anyone in the field would tell you it's close to "solved".

--- 

_If you spot something inaccurate here, feel free to reach out. This is very much a "learning in public" post._
