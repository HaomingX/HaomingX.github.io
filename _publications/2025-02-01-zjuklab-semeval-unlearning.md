---
title: "ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging"
collection: publications
category: conferences
permalink: /publication/2025-02-01-zjuklab-semeval-unlearning
excerpt: 'This paper presents our approach to SemEval-2025 Task 4, which focuses on unlearning in semantic understanding. We propose a model merging strategy that consolidates alternative model versions to enforce effective knowledge removal.'
date: 2025-02-01
venue: 'SemEval-2025'
paperurl: 'https://arxiv.org/abs/2501.00000'
citation: 'Xu, H., et al. (2025). &quot;ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging.&quot; <i>SemEval-2025</i>.'
---

SemEval-2025 Task 4 addresses the challenge of machine unlearning in the context of semantic understanding tasks. This task requires systems to effectively remove specific semantic knowledge while maintaining performance on general tasks.

Our approach, **ZJUKLAB**, leverages model merging techniques to achieve effective unlearning. The core idea is to train multiple model variants with different unlearning targets, then merge these models in a way that reinforces the removal of unwanted knowledge while preserving desired capabilities.

We develop a novel merging strategy that:
- Identifies complementary model components through gradient analysis
- Applies weighted merging with attention to semantic similarity
- Validates unlearning effectiveness through comprehensive evaluation metrics

Our system achieved top performance in the task evaluation, demonstrating the effectiveness of model merging as a strategy for machine unlearning in semantic tasks.
