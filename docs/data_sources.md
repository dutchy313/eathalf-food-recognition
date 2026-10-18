# Data sources and provenance

## Historical experiment

The author identifies the source as a Kaggle Food-101 computer vision dataset. The notebook opens a local archive named `Food_101.zip`; it does not record its Kaggle URL, uploader, version, or checksum. These details remain unresolved. Add the exact Kaggle dataset URL and downloaded version when available; no listing is guessed here.

The original notebook records 16,256 images in 17 categories: 256 apple-pie images and 1,000 for every other class. This describes the local copy, not an intrinsic imbalance in original Food-101. Investigate whether the apple-pie folder was incomplete before treating augmentation as the solution.

## Original benchmark

- Original creators: Lukas Bossard, Matthieu Guillaumin, Luc Van Gool, ETH Zurich.
- Homepage: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
- Paper: *Food-101 – Mining Discriminative Components with Random Forests*, ECCV 2014.
- Original scope: 101 categories and 101,000 images; 750 training and 250 test images per class.
- The original training images intentionally contain some label noise.

This repository uses only 17 selected categories and a custom 70/15/15 split. It does not implement the official benchmark split. Scores must be labelled accordingly and should not be compared directly with full Food-101 benchmark scores.

## Reproducibility and rights

Keep the original downloaded archive privately with its checksum and download date. Record the Kaggle URL/version, file exclusions, class counts, and any manual changes in your experiment log. The corrected preparation command stores relative-path manifests, pixel hashes, unreadable-image logs, and duplicate exclusions; it does not redistribute images.

Obtain images from an authorized source and retain its notices. A Kaggle listing is not itself a grant of rights to all underlying photographs. The repository AGPL license covers the project's code and documentation, not third-party images or weights. Review source terms before redistributing any image examples or using the data in a product.

```bibtex
@inproceedings{bossard14,
  title={Food-101 -- Mining Discriminative Components with Random Forests},
  author={Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle={European Conference on Computer Vision},
  year={2014}
}
```
