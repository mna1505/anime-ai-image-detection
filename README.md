# Anime AI Detection

### Can AI tell the difference?

Research project for detecting **AI-generated anime illustrations** and
**human-created artwork** using deep learning.

---

## About

Generative AI has made anime illustration generation increasingly accessible,
making the visual distinction between human-made and AI-generated artwork
more difficult.

This research investigates how well different deep learning architectures
can distinguish between the two classes, especially when the model encounters
data from sources that were not used during training.

The study compares four architectures:

- **MobileNetV3-Large**
- **ResNet50**
- **EfficientNetV2-S**
- **ViT-B/16**

The experiments evaluate not only classification performance, but also
generalization and inference efficiency.

---

## Research Focus

The experiments are organized into five training stages:

```text
Baseline
   ↓
Level 1 — Robustness Augmentation
   ↓
Level 2 — Training Regularization
   ↓
Level 3 — Discriminative Learning Rates
   ↓
Level 4 — Sharpness-Aware Minimization (SAM)
