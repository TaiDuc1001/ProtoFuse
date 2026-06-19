# ProtoFuse: Equations and Process

## 1. Objective

ProtoFuse is a training-free adaptation procedure for few-shot image classification with CLIP. Its goal is to improve zero-shot text prototypes by incorporating a small labeled support set, while avoiding gradient-based training.

The central idea is simple:

$$
P_c(\alpha) = \operatorname{norm}\big((1 - \alpha)T_c + \alpha V_c\big)
$$

where $T_c$ is the text prototype of class $c$, $V_c$ is the visual prototype estimated from few-shot support images, and $\alpha$ controls how much the final classifier trusts the support set.

## 2. Problem Setup

Let there be $C$ classes. For each class $c \in \{1, \dots, C\}$, we have a small labeled support set:

$$
S_c = \{(I_i, y_i) : y_i = c\}
$$

The model also receives an evaluation set:

$$
D_{eval} = \{(I_j, y_j)\}_{j=1}^{N}
$$

The CLIP image encoder and text encoder are frozen. ProtoFuse does not update model weights, prompts, adapters, or any trainable parameters.

## 3. Notation

The main symbols are:

- $E_I(\cdot)$: frozen CLIP image encoder.
- $E_T(\cdot)$: frozen CLIP text encoder.
- $T_c \in \mathbb{R}^{d}$: normalized text prototype for class $c$.
- $x_i \in \mathbb{R}^{d}$: normalized image embedding of support image $I_i$.
- $V_c \in \mathbb{R}^{d}$: normalized visual prototype for class $c$.
- $P_c(\alpha) \in \mathbb{R}^{d}$: fused prototype for class $c$.
- $\alpha \in [0, 1]$: fusion coefficient.

Here, $\operatorname{norm}(z)$ means L2 normalization:

$$
\operatorname{norm}(z) = \frac{z}{\|z\|_2}
$$

## 4. Text Prototype Construction

For each class name $a_c$, a natural-language prompt $p_c$ is formed. The class text prototype is:

$$
T_c = \operatorname{norm}(E_T(p_c))
$$

Stacking all class text prototypes gives:

$$
T = [T_1; T_2; \dots; T_C] \in \mathbb{R}^{C \times d}
$$

These text prototypes are the zero-shot classifier. If no visual information is used, prediction is performed by comparing an image embedding against $T$.

## 5. Support Image Embeddings

For each support image $I_i$, ProtoFuse extracts a CLIP image embedding:

$$
z_i = E_I(I_i)
$$

Then it normalizes the feature:

$$
x_i = \operatorname{norm}(z_i)
$$

For class $c$, the support image embeddings are:

$$
X_c = \{x_i : y_i = c\}
$$

## 6. Text-Weighted Visual Prototype

A naive visual prototype would average all support embeddings in a class. ProtoFuse instead gives higher weight to support images that are more aligned with the class text prototype.

For every support feature $x_i \in X_c$, compute its clipped similarity to the text prototype:

$$
s_i = \max(x_i^\top T_c, 0)
$$

The weight is:

$$
w_i = \frac{s_i}{\sum_{x_j \in X_c} s_j}
$$

If the denominator is approximately zero, ProtoFuse falls back to uniform weights:

$$
w_i = \frac{1}{|X_c|}
$$

The visual prototype for class $c$ is then:

$$
V_c = \operatorname{norm}\left(\sum_{x_i \in X_c} w_i x_i\right)
$$

This produces one visual prototype per class:

$$
V = [V_1; V_2; \dots; V_C] \in \mathbb{R}^{C \times d}
$$

## 7. Prototype Fusion

ProtoFuse combines the text prototype and visual prototype through a convex interpolation followed by normalization:

$$
P_c(\alpha) = \operatorname{norm}\big((1 - \alpha)T_c + \alpha V_c\big)
$$

Interpretation:

- $\alpha = 0$ gives the original zero-shot CLIP classifier.
- $\alpha = 1$ gives a purely support-set visual classifier.
- $0 < \alpha < 1$ balances semantic prior and few-shot visual evidence.

The candidate set of alpha values is a uniform grid:

$$
\mathcal{A} = \left\{0, \frac{1}{K-1}, \frac{2}{K-1}, \dots, 1\right\}
$$

where $K$ is the number of alpha steps.

## 8. Alpha Selection for Multi-Shot Support

When every class has at least two support images, ProtoFuse selects $\alpha$ using a leave-one-out calibration process.

Let:

$$
k = \min_c |X_c|
$$

For each hold-out index $r \in \{1, \dots, k\}$, one support sample from each class is held out. The held-out feature for class $c$ is:

$$
h_c^{(r)} = x_{c,r}
$$

The visual prototype is rebuilt without the held-out sample:

$$
V_c^{(-r)} = \operatorname{norm}\left(\sum_{x_i \in X_c \setminus \{x_{c,r}\}} w_i^{(-r)}x_i\right)
$$

For each candidate $\alpha$, the temporary fused prototype is:

$$
P_c^{(-r)}(\alpha) = \operatorname{norm}\big((1 - \alpha)T_c + \alpha V_c^{(-r)}\big)
$$

The text-only prediction for the held-out sample is:

$$
\hat{y}_{c,text}^{(r)} = \arg\max_j (h_c^{(r)})^\top T_j
$$

The fused prediction is:

$$
\hat{y}_{c,fuse}^{(r)}(\alpha) = \arg\max_j (h_c^{(r)})^\top P_j^{(-r)}(\alpha)
$$

ProtoFuse measures whether fusion rescues or damages the text-only prediction.

A rescue occurs when text-only classification is wrong but fused classification is correct:

$$
R_c^{(r)}(\alpha) = \mathbb{1}\left[\hat{y}_{c,text}^{(r)} \ne c \;\land\; \hat{y}_{c,fuse}^{(r)}(\alpha) = c\right]
$$

A damage occurs when text-only classification is correct but fused classification becomes wrong:

$$
D_c^{(r)}(\alpha) = \mathbb{1}\left[\hat{y}_{c,text}^{(r)} = c \;\land\; \hat{y}_{c,fuse}^{(r)}(\alpha) \ne c\right]
$$

The alpha score is:

$$
\operatorname{score}(\alpha) = \sum_{r=1}^{k}\sum_{c=1}^{C} R_c^{(r)}(\alpha) - \sum_{r=1}^{k}\sum_{c=1}^{C} D_c^{(r)}(\alpha)
$$

The selected value is:

$$
\alpha^* = \arg\max_{\alpha \in \mathcal{A}} \operatorname{score}(\alpha)
$$

The final class prototype is then:

$$
P_c^* = P_c(\alpha^*)
$$

## 9. Alpha Selection for One-Shot Support

When each class has only one support image, leave-one-out calibration is not possible. ProtoFuse therefore uses a centroid-mix calibration process.

The purpose is to simulate difficult class-confusion cases by mixing each class prototype with a nearby class prototype.

For each class $c$, a neighbor class $n(c)$ is selected by prototype similarity:

$$
n(c) = \arg\max_{j \ne c} \operatorname{sim}(c, j)
$$

The neighbor is selected using visual-prototype similarity:

$$
\operatorname{sim}(c,j) = V_c^\top V_j
$$

For a small mixing coefficient $\beta$, ProtoFuse creates a pseudo feature:

$$
q_c(\beta) = \operatorname{norm}\big((1 - \beta)V_c + \beta V_{n(c)}\big)
$$

The text-only prediction is:

$$
\hat{y}_{c,text}(\beta) = \arg\max_j q_c(\beta)^\top T_j
$$

The fused prediction is:

$$
\hat{y}_{c,fuse}(\alpha, \beta) = \arg\max_j q_c(\beta)^\top P_j(\alpha)
$$

For each neighbor mode and each $\beta$, ProtoFuse computes a rescue-minus-damage curve:

$$
\operatorname{score}_{\beta}(\alpha) = \sum_{c=1}^{C}\mathbb{1}\left[\hat{y}_{c,text}(\beta) \ne c \land \hat{y}_{c,fuse}(\alpha,\beta)=c\right]
$$

$$
\quad - \sum_{c=1}^{C}\mathbb{1}\left[\hat{y}_{c,text}(\beta)=c \land \hat{y}_{c,fuse}(\alpha,\beta)\ne c\right]
$$

The score curve is normalized:

$$
y(\alpha) = \frac{\operatorname{score}_{\beta}(\alpha) - \min_{a \in \mathcal{A}}\operatorname{score}_{\beta}(a)}{\max_{a \in \mathcal{A}}\operatorname{score}_{\beta}(a) - \min_{a \in \mathcal{A}}\operatorname{score}_{\beta}(a)}
$$

The knee score is:

$$
\operatorname{knee}(\alpha) = y(\alpha) - \alpha
$$

The candidate alpha for this curve is:

$$
\alpha_{knee} = \arg\max_{\alpha \in \mathcal{A}} \operatorname{knee}(\alpha)
$$

To avoid selecting weak curves, ProtoFuse also measures curve quality:

$$
\operatorname{quality} = \operatorname{knee}(\alpha_{knee}) \cdot \frac{\max_{a \in \mathcal{A}}\operatorname{score}_{\beta}(a) - \min_{a \in \mathcal{A}}\operatorname{score}_{\beta}(a)}{C}
$$

The final one-shot $\alpha^*$ is the knee alpha from the highest-quality curve. If no curve has positive quality, ProtoFuse falls back to:

$$
\alpha^* = 0
$$

## 10. Classification

After selecting $\alpha^*$, the final prototype for every class is:

$$
P_c^* = \operatorname{norm}\big((1 - \alpha^*)T_c + \alpha^*V_c\big)
$$

For an evaluation image $I$, its normalized embedding is:

$$
x = \operatorname{norm}(E_I(I))
$$

The class logit is cosine similarity to the fused prototype:

$$
\ell_c = x^\top P_c^*
$$

The predicted class is:

$$
\hat{y} = \arg\max_c \ell_c
$$

## 11. Complete Process

The ProtoFuse process is:

1. Build one text prompt for every class.
2. Encode each prompt with the frozen CLIP text encoder.
3. Normalize the text embeddings to obtain $T_c$.
4. Select $k$ labeled support images per class.
5. Encode support images with the frozen CLIP image encoder.
6. Normalize image embeddings to obtain $x_i$.
7. For each class, compute text-weighted visual prototype $V_c$.
8. Sweep candidate values of $\alpha$ on a uniform grid.
9. Select $\alpha^*$ by leave-one-out calibration if support is multi-shot.
10. Select $\alpha^*$ by centroid-mix knee calibration if support is one-shot.
11. Fuse text and visual prototypes using $\alpha^*$.
12. Classify evaluation images by nearest fused prototype under cosine similarity.
13. Report classification metrics.

## 12. Core Intuition

ProtoFuse is built around the tension between two sources of information.

The text prototype $T_c$ is semantically strong and stable, but it may be too generic for fine-grained classes. The visual prototype $V_c$ captures the actual support images, but it can be noisy when only a few examples are available.

The fusion equation:

$$
P_c(\alpha) = \operatorname{norm}\big((1 - \alpha)T_c + \alpha V_c\big)
$$

uses $\alpha$ to decide how much support-set evidence should override the original CLIP text prior.

The alpha selection procedure is designed to choose fusion only when it creates more corrections than regressions:

$$
\operatorname{score}(\alpha) = \operatorname{rescues}(\alpha) - \operatorname{damages}(\alpha)
$$

Thus, ProtoFuse is not simply averaging text and image prototypes. It calibrates the fusion strength according to whether visual evidence improves decision boundaries without destroying correct text-based predictions.
