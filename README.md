# Annotations on a Budget: Leveraging Geo-Data Similarity to Balance Model Performance and Annotation Cost

[Paper](https://arxiv.org/pdf/2403.07687.pdf)
[[ACL Anthology page]](TODO)
[Poster](COLING.BudgetAnnotations.2024.pdf)


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
@inproceedings{ignat-etal-2024-annotations-budget,
    title = "Annotations on a Budget: Leveraging Geo-Data Similarity to Balance Model Performance and Annotation Cost",
    author = "Ignat, Oana  and
      Bai, Longju  and
      Nwatu, Joan C.  and
      Mihalcea, Rada",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italy",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.112",
    pages = "1239--1259",
    abstract = "Current foundation models have shown impressive performance across various tasks. However, several studies have revealed that these models are not effective for everyone due to the imbalanced geographical and economic representation of the data used in the training process. Most of this data comes from Western countries, leading to poor results for underrepresented countries. To address this issue, more data needs to be collected from these countries, but the cost of annotation can be a significant bottleneck. In this paper, we propose methods to identify the data to be annotated to balance model performance and annotation costs. Our approach first involves finding the countries with images of topics (objects and actions) most visually distinct from those already in the training datasets used by current large vision-language foundation models. Next, we identify countries with higher visual similarity for these topics and show that using data from these countries to supplement the training data improves model performance and reduces annotation costs. The resulting lists of countries and corresponding topics are made available at https://github.com/MichiganNLP/visual{\_}diversity{\_}budget.",
}
```
