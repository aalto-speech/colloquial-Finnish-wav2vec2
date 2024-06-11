# Colloquial Finnish wav2vec2
Scripts for training colloquial Finnish wav2vec 2.0 models

## Pre-trained and fine-tuned models

Model | Labeled Data, h | DEV WER, % | TEST WER, %
|---|---|---|---
[Wav2Vec 2.0 Base VP-Finnish](https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_base_fi_v2.pt) | N/A | N/A | N/A
[Wav2Vec 2.0 Base VP-Finnish](https://huggingface.co/GetmanY1/wav2vec2-base-fi-voxpopuli-v2-100h) | 100 | 29.35 | 31.90
[Wav2Vec 2.0 Base VP-Finnish](https://zenodo.org/doi/10.5281/zenodo.11571810) | 1500 | 22.18 | 24.43
[Wav2Vec 2.0 Base LP (PT from scratch)](https://zenodo.org/doi/10.5281/zenodo.11572683) | N/A | N/A | N/A
[Wav2Vec 2.0 Base LP (PT from scratch)](https://huggingface.co/GetmanY1/wav2vec2-base-fi-lp-from-scratch-100h) | 100 | 26.40 | 28.92
[Wav2Vec 2.0 Base LP (PT from scratch)](https://zenodo.org/doi/10.5281/zenodo.11572956) | 1500 | 21.61 | 24.35
[Wav2Vec 2.0 Base LP (continued PT)](https://zenodo.org/doi/10.5281/zenodo.11573133) | N/A | N/A | N/A
[Wav2Vec 2.0 Base LP (continued PT)](https://huggingface.co/GetmanY1/wav2vec2-base-fi-lp-cont-pt-100h) | 100 | 22.49 | 24.95
[Wav2Vec 2.0 Base LP (continued PT)](https://zenodo.org/doi/10.5281/zenodo.11573213) | 1500 | 17.38 | 19.65
[Wav2Vec 2.0 Large VP-Uralic](https://dl.fbaipublicfiles.com/voxpopuli/models/wav2vec2_large_uralic_v2.pt) | N/A | N/A | N/A
[Wav2Vec 2.0 Large VP-Uralic](https://huggingface.co/GetmanY1/wav2vec2-large-uralic-voxpopuli-v2-100h) | 100 | 21.02 | 22.98
[Wav2Vec 2.0 Large VP-Uralic](https://zenodo.org/doi/10.5281/zenodo.11573577) | 1500 | 19.14 | 20.49
[Wav2Vec 2.0 Large LP (PT from scratch)](https://zenodo.org/doi/10.5281/zenodo.11573671) | N/A | N/A | N/A
[Wav2Vec 2.0 Large LP (PT from scratch)](https://huggingface.co/GetmanY1/wav2vec2-large-fi-lp-from-scratch-100h) | 100 | 21.66 | 23.85
[Wav2Vec 2.0 Large LP (PT from scratch)](https://zenodo.org/doi/10.5281/zenodo.11573886) | 1500 | 17.54 | 19.26
[Wav2Vec 2.0 Large LP (continued PT)](https://zenodo.org/doi/10.5281/zenodo.11573973) | N/A | N/A | N/A
[Wav2Vec 2.0 Large LP (continued PT)](https://huggingface.co/GetmanY1/wav2vec2-large-fi-lp-cont-pt-100h) | 100 | 22.49 | 24.95
[Wav2Vec 2.0 Large LP (continued PT)](https://zenodo.org/doi/10.5281/zenodo.11574055) | 1500 | 16.24 | 18.04

More details on the models are available in the [paper](TODO).
The models are also available at [Huggingface Hub](https://huggingface.co/collections/GetmanY1/colloquial-finnish-wav2vec2-665f0d692c7800b0d999920d)

## Pre-training the models

The scripts shared in this repository are adapted to the AMD hardware of the [LUMI supercomputer](https://www.lumi-supercomputer.eu/). To train a wav2vec 2.0 Base model, run

```sbatch /scripts/pretraining/fairseq_train_multinode_w2v2_B_512gpus.sh
```

Note: you can simulate 512 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 512/k

## Fine-tuning the models with CTC

To fine-tune a wav2vec 2.0 Base model using Fairseq, run

```sbatch scripts/finetuning/fairseq_finetune_multinode_w2v2_B_128gpus_full.sh
```

Note: you can simulate 128 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 128/k

## Fine-tuning the models with CTC using 🤗Transformers

To fine-tune a wav2vec 2.0 Base model using Huggingface Transformers, run

```sbatch scripts/finetuning/huggingface_finetune_multinode_w2v2_B_8gpus_full.sh
```

## Citation

If you use our models or scripts, please cite our article as:

```bibtex
@inproceedings{getman24a_interspeech,
  author={Yaroslav Getman and Tamas Grosz and Mikko Kurimo},
  title={{What happens in continued pre-training? Analysis of self-supervised speech
models with continued pre-training for colloquial Finnish ASR}},
  year=2024,
  booktitle={Proc. INTERSPEECH 2024},
  pages={XX--XX},
  doi={XXXX},
  issn={XXXX-XXXX}
}
```
