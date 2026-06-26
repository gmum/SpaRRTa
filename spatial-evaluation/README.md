# SpaRRTa — Spatial Evaluation

This directory covers the **evaluation / probing** half of the SpaRRTa project: measuring whether
frozen Visual Foundation Models (VFMs) encode the **spatial relations between objects** in a scene.
The companion [`unreal-scene-gen/`](../unreal-scene-gen) directory covers how the synthetic scenes
are rendered.

> 📦 **The full evaluation code lives in a dedicated repository:**
> **➡️ https://github.com/turhancan97/SpaRRTa**
>
> This page is an overview; clone that repo to run the experiments.

<p align="center">
  <img src="https://github.com/user-attachments/assets/fd855699-25bd-42a0-88ec-6aa7da3a47ee" alt="SpaRRTa teaser" width="100%">
</p>

## What it does

SpaRRTa is a 4-way classification task — **Front / Back / Left / Right** — predicting the relative
direction of a *target* object with respect to a *reference* object, in two settings:

- **Egocentric** — directions defined from the camera's viewpoint.
- **Allocentric** — directions defined from a human figure's viewpoint (requires perspective-taking).

A frozen backbone produces patch/CLS tokens and only a lightweight **probe head** is trained
(linear / attention-based). The evaluation suite supports **15 backbones** from the paper
(DINO, DINOv2/v3, MAE, CroCo, VGGT, SPA, CLIP, DeiT, MaskFeat …), leave-one-environment-out
transfer, sim-to-real (lego) evaluation, and an attention-analysis pipeline.

## Resources

- 📄 **Paper:** [arXiv:2601.11729](https://arxiv.org/abs/2601.11729)
- 💻 **Evaluation code:** https://github.com/turhancan97/SpaRRTa
- 🚀 **Live demo:** https://huggingface.co/spaces/turhancan97/SpaRRTa-demo
- 🧩 **Dataset (synthetic):** https://huggingface.co/datasets/turhancan97/SpaRRTa
- 🧱 **Real-world (lego) split:** https://huggingface.co/datasets/turhancan97/SpaRRTa-Lego
- 🔬 **Attention-analysis split:** https://huggingface.co/datasets/turhancan97/SpaRRTa-Attention

## Quick start

```bash
git clone https://github.com/turhancan97/SpaRRTa
cd SpaRRTa
conda create -n sparrta python=3.9 --yes && conda activate sparrta
# install PyTorch for your CUDA version (see https://pytorch.org), then:
pip install -e .

# point the code at the data (download from the Hugging Face links above)
export SPARRTA_DATA_ROOT=/path/to/sparrta/unreal

# train an attention probe on DINO features (egocentric, forest)
python train.py \
  backbone=dino_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.EfficientProbing \
  dataset.perspective=camera \
  environment=forest
```

See the [full README](https://github.com/turhancan97/SpaRRTa#readme) for the backbone table,
data layout, transfer protocols, the attention-analysis suite, and how to add your own model.

## Citation

```bibtex
@article{kargin2026sparrta,
  title   = {SpaRRTa: A Synthetic Benchmark for Evaluating Spatial Intelligence in Visual Foundation Models},
  author  = {Karg{\i}n, Turhan Can and Jasi{\'n}ski, Wojciech and Pardyl, Adam and Zieli{\'n}ski, Bartosz and Przewi{\k{e}}{\'z}likowski, Marcin},
  journal = {arXiv preprint arXiv:2601.11729},
  year    = {2026}
}
```
