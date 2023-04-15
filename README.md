# Re Caps
The task of relation extraction focuses on finding relational information between entities in a text. The input text can be one sentence or a document. Relational facts are highly semantic information that needs reasoning skills to discover. Therefore, the model must be capable of processing the text's semantic concepts. 

In this thesis, we explored the potential of the Caps-net in extracting high-level semantic information, capturing the relational information from the text, and the ability to overcome the imbalance problem of the data set.
Our dataset was a complex data set with a severe imbalance and a relatively large number of potential entity-pair per document. The large number of entity pairs along with the complexity needed for solving this task, make the whole proposed model computationally expensive. In addition to trying to improve the model's performance on the dataset, we tried to reduce the memory and time complexity of the proposed model.

In this work, we studied the Caps-Net [[1]](#1) capability for the relation extraction task. We customized the Caps-Net[[1]](#1) for this task and defined a new loss function for the Caps-Net[[1]](#1).
First, we train our model only on positive data to see if the caps net can capture relational information or not. after getting the result we tried to apply the model on a fraction of negative data and all of the positive data in a way that negative samples consist of 50\% of the whole samples. Our model could not generalize on the negative data, so we trained a binary classifier to discriminate between negative and positive samples. 

The DocRED[[2]](#2) data set was quite challenging and our models often took a long time to train. Due to the severe imbalance in the data, a small batch could not represent the data distribution properly. On the other hand, for large batch sizes, each document's potential entity-pair count was a large number and needed a significant amount of memory. We used gradient accumulation and mixed precision to solve this problem. We achieved an F1 score of 42.8 on the test set.


## References
<a id="1">[1]</a> Sabour, Sara, Nicholas Frosst, and Geoffrey E Hinton (2017). Dynamic Routing Between Capsules.doi: 10.48550/ARXIV.1710.09829. url: https://arxiv.org/abs/1710.09829.

<a id="2">[2]</a> Yuan Yao et al. DocRED: A Large-Scale Document-Level Relation Extraction Dataset. 2019. arXiv: 1906.
06127 [cs.CL].
