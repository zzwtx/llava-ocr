---
dataset_info:
  features:
  - name: dataset
    dtype: string
  - name: question
    dtype: string
  - name: question_type
    dtype: string
  - name: answer
    sequence: string
  - name: image
    dtype: image
  splits:
  - name: test
    num_bytes: 85534416.0
    num_examples: 1000
  download_size: 67576988
  dataset_size: 85534416.0
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---
[Github](https://github.com/Yuliang-Liu/MultimodalOCR)|[Paper](https://arxiv.org/abs/2305.07895)


OCRBench has been accepted by [Science China Information Sciences](https://link.springer.com/article/10.1007/s11432-024-4235-6).




