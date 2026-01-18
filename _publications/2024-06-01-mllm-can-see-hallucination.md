---
title: "MLLM Can See? Dynamic Correction Decoding for Hallucination Mitigation"
collection: publications
category: conferences
permalink: /publication/2024-06-01-mllm-can-see-hallucination
excerpt: 'This paper introduces a dynamic correction decoding strategy for multimodal large language models (MLLMs) that leverages visual information to detect and correct hallucinations during text generation, significantly reducing factual errors.'
date: 2024-06-01
venue: 'arXiv preprint'
paperurl: 'https://arxiv.org/abs/2406.00000'
citation: 'Xu, H., et al. (2024). &quot;MLLM Can See? Dynamic Correction Decoding for Hallucination Mitigation.&quot; <i>arXiv preprint arXiv:2406.00000</i>.'
---

Multimodal Large Language Models (MLLMs) have shown remarkable capabilities in vision-language tasks. However, they often suffer from hallucinations—generating text that contradicts or invents information not present in the visual input. This is particularly problematic in applications requiring factual accuracy.

This paper presents **Dynamic Correction Decoding (DCD)**, a novel decoding strategy that actively leverages visual information to mitigate hallucinations. DCD operates in two phases: (1) hallucination detection through cross-modal consistency checking, and (2) dynamic correction by generating alternative tokens when inconsistencies are detected.

Key contributions include:
- A real-time hallucination detection mechanism based on vision-language alignment scores
- A correction module that generates contextually appropriate alternatives during decoding
- Evaluation on multiple vision-language benchmarks showing 40-50% reduction in factual errors

Experimental results demonstrate that DCD significantly improves the reliability of MLLMs while maintaining generation fluency, making it suitable for deployment in production systems where accuracy is critical.
