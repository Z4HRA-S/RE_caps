import torch
import transformers as ppb
from torch import nn
from itertools import product
from caps_net import CapsNet


class Model(nn.Module):
    def __init__(self, len_tokenizer, device, use_negative=False):
        self.use_negative = use_negative
        super(Model, self).__init__()
        self.embedding_model = ppb.BertModel.from_pretrained("bert-base-uncased",
                                                             hidden_dropout_prob=0.2)
        self.embedding_model.resize_token_embeddings(len_tokenizer)
        self.embedding_model.to(device)
        self.caps_net = CapsNet(num_class=96, device=device)
        self.caps_net.to(device)
        self.lstm = torch.nn.LSTM(768, 384, 2, bidirectional=True, device=device)
        self.device = device

    def forward(self, x):
        input_ids = x["input_ids"].to(self.device)
        attention_mask = x["attention_mask"].to(self.device)

        embedded_doc = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True)

        feature_set, labels = self.extract_feature(embedded_doc, x)
        output, (hn, cn) = self.lstm(feature_set)
        output = self.caps_net(output)
        return output, labels

    def get_pred(self, output):
        norms = output.squeeze().norm(dim=-1)
        # indices=norms.max(dim=-1)[1]
        # preds = torch.nn.functional.one_hot(indices,num_classes=96)
        preds = norms.gt(0.5).float()
        return preds

    def extract_feature(self, embedded_doc, x):
        entities = x["entity_list"]
        entity_list, attentions = self.aggregate_entities(entities, embedded_doc)
        cls_tokens = embedded_doc.last_hidden_state[:, 0]

        feature_set = []
        labels = []

        for i, (entities_embedding, attention_vector) in enumerate(zip(entity_list, attentions)):
            label = x["label"][i].to(self.device)
            if self.use_negative:
                all_possible_ent_pair = self.all_possible_pair(len(entities_embedding))
            else:
                all_possible_ent_pair = self.labeled_pair(label)

            feature_set.extend(
                [
                    torch.stack([
                        cls_tokens[i],
                        entities_embedding[h],
                        entities_embedding[t],
                        attention_vector[h] * attention_vector[t]
                    ])
                    for h, t in all_possible_ent_pair]
            )

            labels.extend([label[i][j] for i, j in all_possible_ent_pair])
        feature_set = torch.stack(feature_set).to(self.device)
        labels = torch.stack(labels)
        return feature_set, labels

    def all_possible_pair(self, num_ent: int):
        return list(
            filter(lambda x: x[1] != x[0],
                   product(range(num_ent),
                           range(num_ent)))
        )

    def labeled_pair(self, labels):
        num_ent = labels.size()[0]
        labeled = [(i, j) for i in range(num_ent) for j in range(num_ent) if torch.sum(labels[i][j]) > 0]
        return labeled

    def aggregate_entities(self, vertexSet_list: list, embedded_doc_list):
        batch = []
        context_att = []
        embedding = embedded_doc_list.last_hidden_state

        # batch_size * token * token
        last_attention_lr = torch.mean(embedded_doc_list.attentions[-1], dim=1)

        for vertexSet, embedded_doc, attention in zip(vertexSet_list, embedding, last_attention_lr):
            mention_positions = [list(filter(lambda x: x[0] <= 512 and x[1] <= 512, ent)) for ent in vertexSet]
            mention_positions = [list(filter(lambda x: len(x) > 0, mnt)) for mnt in mention_positions]
            mention_positions = list(filter(lambda x: len(x) > 0, mention_positions))

            context = [
                torch.mean(
                    torch.matmul(
                        torch.concat([attention[pos[0] + 1:pos[1] + 1] for pos in mnt]), embedded_doc),
                    dim=0)
                for mnt in mention_positions]
            #  doc contains aggregated entity embedding, in shape(ent, 768)
            doc = [torch.logsumexp(torch.concat([embedded_doc[pos[0] + 1:pos[1] + 1] for pos in mnt]), dim=0)
                   for mnt in mention_positions]

            doc = torch.stack(doc)
            batch.append(doc)

            context = torch.stack(context)
            context_att.append(context)
        return batch, context_att
