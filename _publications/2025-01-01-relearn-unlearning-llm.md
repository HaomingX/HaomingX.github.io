---
title: "Relearn: Unlearning via Learning for Large Language Models"
collection: publications
category: conferences
permalink: /publication/2025-01-01-relearn-unlearning-llm
excerpt: 'This paper introduces Relearn, a novel framework for machine unlearning in large language models that enables effective removal of specific knowledge while preserving model performance through a learning-based approach.'
date: 2025-01-01
venue: 'arXiv preprint'
paperurl: 'https://arxiv.org/abs/2408.15168'
citation: 'Xu, H., et al. (2025). &quot;Relearn: Unlearning via Learning for Large Language Models.&quot; <i>arXiv preprint arXiv:2408.15168</i>.'
---

Large language models (LLMs) have demonstrated remarkable capabilities across various tasks. However, the need to remove specific information from trained models—whether for privacy, legal compliance, or model refinement—has become increasingly important. Traditional fine-tuning approaches often fail to completely erase targeted knowledge while maintaining overall model performance.

This paper presents **Relearn**, a novel framework for machine unlearning in LLMs that leverages a learning-based approach. Unlike existing methods that rely solely on gradient-based unlearning, Relearn employs a dual-phase process: (1) targeted unlearning to reduce retention of specific knowledge, and (2) knowledge restoration to preserve general capabilities.

Our experiments demonstrate that Relearn achieves significantly better unlearning effectiveness compared to baseline methods, reducing unwanted knowledge retention by over 60% while maintaining task performance within 2% of the original model. The framework is scalable and applicable to various LLM architectures, making it a practical solution for knowledge removal in production systems.
