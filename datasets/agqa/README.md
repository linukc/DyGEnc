# [AGQA](https://cs.stanford.edu/people/ranjaykrishna/agqa/)

Benchmark for Compositional Spatio-Temporal Reasoning.

## Download

From the [project site](https://cs.stanford.edu/people/ranjaykrishna/agqa/) download:
1. [Scene graphs](https://drive.google.com/uc?export=download&id=1CXU0tWpv-1kkkwkNzpU-BwQoAazPR1kR) (611 Mb)
2. [Balanced AGQA](https://agqa-decomp.cs.washington.edu/data/agqa2/AGQA_balanced.zip) with 2.27M questions (2.2G)
3. [CSV formatted questions](https://agqa-decomp.cs.washington.edu/data/agqa2/csvs.zip) for evaluation (1.3G)
4. Supporting data

You should get following structure:

```text
├── agqa/
    └── data/
        └── AGQA_scene_graphs/
            └── AGQA_train_stsgs.pkl
            └── AGQA_test_stsgs.pkl
        └── ENG.txt
        └── IDX.txt
```

## Citation

```text
@inproceedings{grunde2021agqa,
  title={Agqa: A benchmark for compositional spatio-temporal reasoning},
  author={Grunde-McLaughlin, Madeleine and Krishna, Ranjay and Agrawala, Maneesh},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={11287--11297},
  year={2021}
}
```

```text
@article{grunde2022agqa,
  title={Agqa 2.0: An updated benchmark for compositional spatio-temporal reasoning},
  author={Grunde-McLaughlin, Madeleine and Krishna, Ranjay and Agrawala, Maneesh},
  journal={arXiv preprint arXiv:2204.06105},
  year={2022}
}
```