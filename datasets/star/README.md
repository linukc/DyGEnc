# [STAR](https://bobbywu.com/STAR/)

A Benchmark for Situated Reasoning in Real-World Videos.

## Download

From the [project site](https://github.com/csbobby/STAR_Benchmark) download:
1. [SG+QA](https://github.com/csbobby/STAR_Benchmark?tab=readme-ov-file#question-multiple-choice-answers-and-situation-graphs)
2. [Split json](https://star-benchmark.s3.us-east.cloud-object-storage.appdomain.cloud/Question_Answer_SituationGraph/split_file.json)
3. [Annotations](https://github.com/csbobby/STAR_Benchmark/tree/main/annotations/STAR_classes) for mapping

You should get following structure:

```text
├── star/
    └── data/
        └── mapping_annotations/
            └── action_classes.txt
            └── action_mapping.txt
            └── object_classes.txt
            └── relationship_classes.txt
            └── verb_classes.txt
        └── STAR_train.json
        └── STAR_val.json
        └── split_file.json
```

## Citation

```text
@article{wu2024star,
  title={Star: A benchmark for situated reasoning in real-world videos},
  author={Wu, Bo and Yu, Shoubin and Chen, Zhenfang and Tenenbaum, Joshua B and Gan, Chuang},
  journal={arXiv preprint arXiv:2405.09711},
  year={2024}
}
```
