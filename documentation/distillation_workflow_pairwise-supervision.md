# Distillation Workflow: Pairwise Supervision

![](documentation/figures/pairwise-supervision-workflow.png)

For pairwise supervision you need one or more trained BERT_CAT models to generate the teacher scores with per model with *matchmaker/distillation/teacher-train-scorer.py*, then you can use it directly in train_tsv (if you use only 1 model) or ensemble the scores with *matchmaker/distillation/teacher_scores_ensemble.py* for multiple models used.

Then set the following config to activate the knowledge distillation loss: 

````yaml
train_pairwise_distillation: True
loss: "margin-mse"
````

**Please cite our work as:**
````
@article{hofstaetter2020_crossarchitecture_kd,
      title={Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation}, 
      author={Sebastian Hofst{\"a}tter and Sophia Althammer and Michael Schr{\"o}der and Mete Sertkan and Allan Hanbury},
      year={2020},
      journal={arXiv:2010.02666},
}
````