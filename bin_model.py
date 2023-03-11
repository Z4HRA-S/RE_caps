import torch
import transformers as ppb
from torch import nn
from itertools import product
from binclassifier import BinaryClassifier
from random import shuffle


class Model(nn.Module):
    def __init__(self, len_tokenizer, device, use_negative=False):
        self.use_negative = use_negative
        super(Model, self).__init__()
        self.embedding_model = ppb.BertModel.from_pretrained("bert-base-uncased",
                                                             hidden_dropout_prob=0.2)
        self.embedding_model.resize_token_embeddings(len_tokenizer)
        self.embedding_model.to(device)
        self.bin_classifier = BinaryClassifier()
        self.device = device

    def forward(self, x, test=False):
        input_ids = x["input_ids"].to(self.device)
        attention_mask = x["attention_mask"].to(self.device)

        embedded_doc = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True)

        # batch_size * token * token
        last_attention_lr = torch.mean(embedded_doc.attentions[-1], dim=1)
        embedded_doc = embedded_doc.last_hidden_state
        feature_set, labels, ent_types = self.extract_feature(embedded_doc,last_attention_lr, x, test)
        output = self.bin_classifier(feature_set)
        loss = self.bin_classifier.loss_func(output, labels)
        return nn.Sigmoid()(output), labels, loss

    def extract_feature(self, embedded_doc,last_attention_lr, x, test):
        entities = x["entity_list"]
        entity_list, attentions = self.aggregate_entities(entities, embedded_doc,last_attention_lr)
        cls_tokens = embedded_doc[:, 0]

        feature_set = []
        labels = []
        types = []

        get_local_context = lambda h, t, i: torch.logsumexp(torch.stack(
            [h * t] * 768, dim=1) * embedded_doc[i], dim=0).squeeze()

        for i, (entities_embedding, attention_vector) in enumerate(zip(entity_list, attentions)):
            label = x["label"][i].to(self.device)
            if self.use_negative or test:
                all_possible_ent_pair = self.all_possible_pair(len(entities_embedding))
            else:
                all_possible_ent_pair = self.labeled_pair(label)

            all_possible_ent_pair = list(
                filter(lambda item: item[0] < len(entities_embedding) and item[1] < len(entities_embedding),
                       all_possible_ent_pair))

            feature_set.extend(
                [
                    torch.stack([
                        cls_tokens[i],
                        entities_embedding[h],
                        entities_embedding[t],
                        get_local_context(attention_vector[h], attention_vector[t], i)
                    ])
                    for h, t in all_possible_ent_pair]
            )

            types.extend([[x["types"][i][h], x["types"][i][t]] for h, t in all_possible_ent_pair])

            labels.extend([label[k][j] for k, j in all_possible_ent_pair])
        feature_set = torch.stack(feature_set).to(self.device)
        labels = torch.stack(labels)
        types = nn.functional.one_hot(torch.Tensor(types).to(torch.int64), num_classes=6).sum(dim=-2).to(self.device)
        types = types.to(torch.float32)
        # types = nn.functional.pad(types, (0, 762))
        return feature_set, labels, types

    def all_possible_pair(self, num_ent: int):
        return list(
            filter(lambda x: x[1] != x[0],
                   product(range(num_ent),
                           range(num_ent)))
        )

    def labeled_pair(self, labels):
        num_ent = labels.size()[0]
        labeled = [(i, j) for i in range(num_ent) for j in range(num_ent) if torch.sum(labels[i][j]) > 0]
        un_labeled = list(filter(lambda x: x[1] != x[0], product(range(num_ent), range(num_ent))))
        un_labeled = list(filter(lambda x: x not in labeled, un_labeled))
        shuffle(un_labeled)
        num_unlabeled = len(labeled)
        labeled.extend(un_labeled[:num_unlabeled])
        shuffle(labeled)
        return labeled

    def aggregate_entities(self, vertexSet_list: list, embedded_doc_list, last_attention_lr):
        batch = []
        context_att = []

        for vertexSet, embedded_doc, attention in zip(vertexSet_list, embedded_doc_list, last_attention_lr):
            mention_positions = [list(filter(lambda x: x[0] <= 512 and x[1] <= 512, ent)) for ent in vertexSet]
            mention_positions = [list(filter(lambda x: len(x) > 0, mnt)) for mnt in mention_positions]
            mention_positions = list(filter(lambda x: len(x) > 0, mention_positions))

            context = [
                torch.mean(
                    torch.concat([attention[pos[0] + 1].unsqueeze(dim=0) for pos in mnt], dim=0),
                    dim=0)
                for mnt in mention_positions]
            #  doc contains aggregated entity embedding, in shape(ent, 768)
            doc = [
                torch.logsumexp(torch.concat([embedded_doc[pos[0] + 1].unsqueeze(dim=0) for pos in mnt], dim=0), dim=0)
                for mnt in mention_positions]

            doc = torch.stack(doc)
            batch.append(doc)

            context = torch.stack(context)
            context_att.append(context)
        return batch, context_att
