---
title: Home
description: SpaRRTa - A Synthetic Benchmark for Evaluating Spatial Intelligence in Visual Foundation Models
hide:
  - navigation
  - toc
---

<div class="hero-section">
  <div class="hero-content">
    <h1 class="hero-title">SpaRRTa</h1>
    <p class="hero-subtitle">
      <strong>Spa</strong>tial <strong>R</strong>elation <strong>R</strong>ecognition <strong>Ta</strong>sk<br>
      A Synthetic Benchmark for Evaluating Spatial Intelligence in Visual Foundation Models
    </p>
    
    <div class="hero-badges">
      <a href="https://github.com/gmum/SpaRRTa">
        <img src="https://img.shields.io/github/stars/gmum/SpaRRTa?style=for-the-badge&logo=github&logoColor=white&labelColor=1a1a2e&color=7c4dff" alt="GitHub Stars">
      </a>
      <a href="https://github.com/gmum/SpaRRTa/blob/main/LICENSE">
        <img src="https://img.shields.io/github/license/gmum/SpaRRTa?style=for-the-badge&labelColor=1a1a2e&color=536dfe" alt="License">
      </a>
      <a href="https://arxiv.org/abs/XXXX.XXXXX">
        <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=1a1a2e" alt="arXiv">
      </a>
      <img src="https://img.shields.io/badge/Unreal_Engine-5.5-0E1128?style=for-the-badge&logo=unrealengine&logoColor=white&labelColor=1a1a2e" alt="UE5">
    </div>
    
    <div class="hero-buttons">
      <a href="getting-started/" class="hero-btn hero-btn-primary">
        <span class="twemoji">🚀</span> Get Started
      </a>
      <a href="https://github.com/gmum/SpaRRTa" class="hero-btn hero-btn-secondary" target="_blank">
        <span class="twemoji">📦</span> GitHub
      </a>
      <a href="https://arxiv.org/abs/XXXX.XXXXX" class="hero-btn hero-btn-secondary" target="_blank">
        <span class="twemoji">📄</span> Paper
      </a>
    </div>
    
    <div class="teaser-container">
      <div class="teaser-glow"></div>
      <img src="imgs/main/teaser.png" alt="SpaRRTa Teaser" class="teaser-image">
    </div>
  </div>
</div>

## Abstract

**Visual Foundation Models (VFMs)**, such as DINO and CLIP, exhibit strong semantic understanding but show limited spatial reasoning capabilities, which limits their applicability to embodied systems. Recent work incorporates 3D tasks (such as depth estimation) into VFM training. However, VFM performance remains inconsistent across different tasks, raising the question: **do these models truly have spatial awareness or overfit to specific 3D objectives?**

To address this question, we introduce the **Spatial Relation Recognition Task (SpaRRTa)** benchmark, which evaluates the representations of relative positions of objects across different viewpoints. SpaRRTa can generate an arbitrary number of photorealistic images with diverse scenes and fully controllable object arrangements, along with freely accessible spatial annotations.

<div class="feature-grid">
  <div class="feature-card">
    <div class="feature-icon">🎮</div>
    <div class="feature-title">Unreal Engine 5</div>
    <div class="feature-description">
      Photorealistic synthetic scenes with full control over object placement, camera positions, and environmental conditions.
    </div>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">🔬</div>
    <div class="feature-title">Spatial Reasoning</div>
    <div class="feature-description">
      Evaluates abstract, human-like relational understanding beyond simple depth estimation or metric prediction.
    </div>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">👁️</div>
    <div class="feature-title">Egocentric & Allocentric</div>
    <div class="feature-description">
      Two task variants testing camera-centric and perspective-taking spatial reasoning abilities.
    </div>
  </div>
  
  <div class="feature-card">
    <div class="feature-icon">📊</div>
    <div class="feature-title">Comprehensive Benchmark</div>
    <div class="feature-description">
      Evaluate 13+ VFMs across 5 diverse environments with multiple probing strategies.
    </div>
  </div>
</div>

## Key Statistics

<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-value">5</div>
    <div class="stat-label">Environments</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">13+</div>
    <div class="stat-label">VFMs Evaluated</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">50K+</div>
    <div class="stat-label">Images</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">3</div>
    <div class="stat-label">Probing Methods</div>
  </div>
</div>

## Key Findings

!!! success "Main Results"
    
    1. **Spatial information is patch-level**: Spatial relations are primarily encoded at the patch level and largely obscured by global pooling
    
    2. **3D supervision enriches patch features**: VGGT (3D-supervised) shows improvements only with selective probing, not linear probing
    
    3. **Allocentric reasoning is challenging**: All models struggle with perspective-taking tasks compared to egocentric variants
    
    4. **Environment complexity matters**: Performance degrades significantly in cluttered environments like City scenes

## Environments

<div class="env-gallery">
  <div class="env-card">
    <img src="imgs/scene_examples/winter_town_image.jpg" alt="Winter Town">
    <div class="env-card-overlay">
      <div class="env-card-title">🏔️ Winter Town</div>
    </div>
  </div>
  <div class="env-card">
    <img src="imgs/scene_examples/bridge_image.jpg" alt="Bridge">
    <div class="env-card-overlay">
      <div class="env-card-title">🌉 Bridge</div>
    </div>
  </div>
</div>

[View All Environments →](unreal-scene-generation.md){ .md-button }

## Evaluation Pipeline

<figure markdown>
  ![Pipeline](imgs/main/pipeline.png){ width="100%" }
  <figcaption>The SpaRRTa evaluation pipeline: (1) Set Stage with diverse assets, (2) Set Camera position, (3) Render photorealistic image, (4) Extract ground truth, (5) Run VFM and probe, (6) Calculate accuracy.</figcaption>
</figure>

## Authors

<div class="authors-grid">
  <div class="author-card">
    <div class="author-name">Turhan Can Kargin</div>
    <div class="author-affiliation">Jagiellonian University</div>
    <div class="author-links">
      <a href="mailto:turhancan.kargin@doctoral.uj.edu.pl" title="Email">✉️</a>
      <a href="https://github.com/turhancan97" title="GitHub">💻</a>
    </div>
  </div>
  
  <div class="author-card">
    <div class="author-name">Wojciech Jasiński</div>
    <div class="author-affiliation">Jagiellonian University, AGH</div>
  </div>
  
  <div class="author-card">
    <div class="author-name">Adam Pardyl</div>
    <div class="author-affiliation">Jagiellonian University, IDEAS NCBR</div>
  </div>
  
  <div class="author-card">
    <div class="author-name">Bartosz Zieliński</div>
    <div class="author-affiliation">Jagiellonian University</div>
  </div>
  
  <div class="author-card">
    <div class="author-name">Marcin Przewięźlikowski</div>
    <div class="author-affiliation">Jagiellonian University</div>
  </div>
</div>

## Affiliations

<div class="affiliations">
  <img src="imgs/orgs/uj.png" alt="Jagiellonian University" class="affiliation-logo" title="Jagiellonian University">
  <img src="imgs/orgs/gmum.png" alt="GMUM" class="affiliation-logo" title="GMUM - Group of Machine Learning Research">
  <img src="imgs/orgs/ideas.png" alt="IDEAS NCBR" class="affiliation-logo" title="IDEAS NCBR">
  <img src="imgs/orgs/agh.png" alt="AGH University of Krakow" class="affiliation-logo" title="AGH University of Krakow">
</div>

## Citation

If you find SpaRRTa useful in your research, please cite our paper:

<div class="citation-box">
  <button class="citation-copy-btn">📋 Copy BibTeX</button>
  <pre><code class="language-bibtex">@article{kargin2025sparrta,
  title={SpaRRTa: A Synthetic Benchmark for Evaluating Spatial Intelligence in Visual Foundation Models},
  author={Kargin, Turhan Can and Jasiński, Wojciech and Pardyl, Adam and Zieliński, Bartosz and Przewięźlikowski, Marcin},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}</code></pre>
</div>

## Acknowledgments

This work was supported by the Polish National Science Center and conducted at the Faculty of Mathematics and Computer Science, Jagiellonian University.

---

<div style="text-align: center; margin-top: 2rem;">
  <a href="getting-started/" class="md-button md-button--primary">Get Started with SpaRRTa</a>
  <a href="results/" class="md-button">View Results</a>
</div>

