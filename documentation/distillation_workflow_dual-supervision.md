# Distillation Workflow: Dual (Pairwise + In-batch-negatives) Supervision


![](documentation/figures/dual-supervision.png)

For dual supervision to work you need pairwise scores from teacher models, as described in [distillation_workflow_pairwise-supervision.md](documentation/distillation_workflow_pairwise-supervision.md) and in addition a trained ColBERT model (best trained with those same pairwise teacher scores, and without the MASK augmenting).

The ColBERT teacher used for in-batch-negatives is run in a dynamic subprocess and infers every batch as it comes from the dataloader (this makes it independent from the used dataloader), but usually we use it for TAS-Balanced 

The required config settings are as follows:

````yaml
#
# pairwise supervision (via static scores)
#
train_pairwise_distillation: True
loss: "margin-mse"

#
# in batch teacher (via dynamic teacher)
#
in_batch_negatives: True
in_batch_neg_lambda: 0.75       # here you can scale the loss influence
in_batch_main_pair_lambda: 1    # if set to 0 only use in-batch neg loss
in_batch_neg_loss: "margin-mse" #KLDivTeacherList

dynamic_teacher: True
dynamic_teacher_in_batch_scoring: True

````



**Please cite TAS-Balanced as:**
````
@inproceedings{Hofstaetter2021_tasb_dense_retrieval,
 author = {Sebastian Hofst{\"a}tter and Sheng-Chieh Lin and Jheng-Hong Yang and Jimmy Lin and Allan Hanbury},
 title = {{Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling}},
 booktitle = {Proc. of SIGIR},
 year = {2021},
}
````
