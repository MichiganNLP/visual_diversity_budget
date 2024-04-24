# Annotations on a Budget: Leveraging Geo-Data Similarity to Balance Model Performance and Annotation Cost

[[Paper]](https://arxiv.org/pdf/2403.07687.pdf)
[[ACL Anthology page]](TODO)
[[Poster]](TODO)


This work proposes methods to **identify the data to be annotated, to balance model performance and annotation costs**. 

![Vision-language models work poorly on data from underrepresented countries. This is primarily due to the diverse appearance of topics (objects and actions) across countries (e.g., ``toothbrush''). However, collecting diverse global data is very expensive. As solutions to budget annotations, we propose to (1) annotate the images visually different from the ones in high-resource datasets such as LAION or ImageNet; (2) supplement data from low-resource countries with data from visually similar countries.](task_overview.png)

Vision-language models work poorly on data from underrepresented countries. This is primarily due to the diverse appearance of topics (objects and actions) across countries (e.g., ``toothbrush''). However, **collecting diverse global data is very expensive**. As **solutions to budget annotations**, we propose to: (1) annotate the images visually different from the ones in high-resource datasets such as LAION or ImageNet; (2) supplement data from low-resource countries with data from visually similar countries.

We hope our work contributes to **building more inclusive and affordable vision-language models and datasets** to help democratize AI globally.

For more information, read our [COLING 2024](https://lrec-coling-2024.org/) paper:

[Annotations on a Budget: Leveraging Geo-Data Similarity to Balance Model Performance and Annotation Cost](https://arxiv.org/pdf/2403.07687.pdf)

By [Oana Ignat](https://oanaignat.github.io/), [Longju Bai](https://longjubai.github.io/), [Joan Nwatu](https://anniejoan.github.io/), and
[Rada Mihalcea](https://web.eecs.umich.edu/~mihalcea/).


This repository includes the obtained results.

## Obtained Results

1. The data before and after pre-processing and the **topic mapping** is shown in [data/data_pre-processing.csv](data/data_pre-processing.csv) 

2. The removed **(topic, country) pairs with less than 10 images** are shown in [data/data_removed.csv](data/data_removed.csv)   

3. The **RQ1 answer**, all the (topic, country) pairs that are consistently dissimilar
to the high-resource data are in [data/output_RQ1.csv](data/output_RQ1.csv)

4. The **RQ2 answer**, all the (topic, country) pairs, and their
most similar countries are in [data/output_RQ2.csv](data/output_RQ2.csv)

## Citation

```bibtex
@inproceedings{ignat-etal-2024-budget,
    title = "Annotations on a Budget: Leveraging Geo-Data Similarity to Balance Model Performance and Annotation Cost",
    author = "Ignat, Oana  and
      Bai, Longju  and
      Nwatu, Joan  and
      Mihalcea, Rada",
    booktitle = "TODO",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/TODO",
    pages = "TODO",
    series = {COLING '24}
}
```
