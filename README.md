# Colloquial Finnish wav2vec2
Scripts for training colloquial Finnish wav2vec2 models



## Pre-trained and fine-tuned models

Model | Labeled Data, h | DEV WER, % | TEST WER, %
|---|---|---|---
[Wav2Vec 2.0 Base VP-Finnish](TODO) | N/A | N/A | N/A
[Wav2Vec 2.0 Base VP-Finnish](TODO) | 100 | 29.35 | 31.90
[Wav2Vec 2.0 Base VP-Finnish](TODO) | 1500 | 22.18 | 24.43
[Wav2Vec 2.0 Base LP (PT from scratch)](TODO) | N/A | N/A | N/A
[Wav2Vec 2.0 Base LP (PT from scratch)](TODO) | 100 | 26.40 | 28.92
[Wav2Vec 2.0 Base LP (PT from scratch)](TODO) | 1500 | 21.61 | 24.35
[Wav2Vec 2.0 Base LP (continued PT)](TODO) | N/A | N/A | N/A
[Wav2Vec 2.0 Base LP (continued PT)](TODO) | 100 | 22.49 | 24.95
[Wav2Vec 2.0 Base LP (continued PT)](TODO) | 1500 | 17.38 | 19.65
[Wav2Vec 2.0 Large VP-Uralic](TODO) | N/A | N/A | N/A
[Wav2Vec 2.0 Large VP-Uralic](TODO) | 100 | 21.02 | 22.98
[Wav2Vec 2.0 Large VP-Uralic](TODO) | 1500 | 19.14 | 20.49
[Wav2Vec 2.0 Large LP (PT from scratch)](TODO) | N/A | N/A | N/A
[Wav2Vec 2.0 Large LP (PT from scratch)](TODO) | 100 | 21.66 | 23.85
[Wav2Vec 2.0 Large LP (PT from scratch)](TODO) | 1500 | 17.54 | 19.26
[Wav2Vec 2.0 Large LP (continued PT)](TODO) | N/A | N/A | N/A
[Wav2Vec 2.0 Large LP (continued PT)](TODO) | 100 | 22.49 | 24.95
[Wav2Vec 2.0 Large LP (continued PT)](TODO) | 1500 | 16.24 | 18.04

The models are also available at [Huggingface Hub](https://huggingface.co/collections/GetmanY1/colloquial-finnish-wav2vec2-665f0d692c7800b0d999920d)


## Pre-training the models

TODO

```sbatch ...
```

Note: you can simulate 512 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 512/k

## Fine-tuning the models with CTC

TODO

```sbatch ...
```

Note: you can simulate 128 GPUs by using k GPUs and adding command line parameters (before `--config-dir`)
`distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 128/k

## Fine-tuning the models with CTC using 🤗Transformers

TODO

```sbatch ...
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
