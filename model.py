import torch
import transformers as ppb
from torch import nn
from itertools import product
from collections import Counter
from caps_net import CapsNet


class Model(nn.Module):
    def __init__(self, len_tokenizer, device):
        super(Model, self).__init__()
        self.embedding_model = ppb.BertModel.from_pretrained("bert-base-uncased")
        self.embedding_model.resize_token_embeddings(len_tokenizer)
        self.embedding_model.to(device)
        self.caps_net = CapsNet()
        self.caps_net.to(device)
        self.device = device

    def forward(self, x):
        input_ids = x["input_ids"]
        attention_mask = x["attention_mask"]

        embedded_doc = self.embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True)

        feature_set, labels = self.extract_feature(embedded_doc, x)
        output = [self.caps_net(feature_set[i:i + 20]) for i in range(0, len(feature_set), 20)]
        output = torch.concat(output, dim=0)
        output = torch.flatten(output, 1, -1)
        output = nn.Linear(output.size()[1], 96, device=self.device)(output)
        output = nn.LeakyReLU()(output)
        output = nn.Linear(96, 96, device=self.device)(output)
        output = nn.Sigmoid()(output)
        return output

    def extract_feature(self, embedded_doc, x):
        entity_list = self.aggregate_entities(x["entity_list"], embedded_doc.last_hidden_state)
        cls_tokens = embedded_doc.last_hidden_state[:, 0]

        feature_set = []
        labels = []
        for i, entities_embedding in enumerate(entity_list):
            all_possible_ent_pair = self.all_possible_pair(len(entities_embedding))
            feature_set.extend(
                [
                    torch.stack([
                        cls_tokens[i],
                        entities_embedding[h],
                        entities_embedding[t]
                    ])
                    for h, t in all_possible_ent_pair]
            )

            label = x["label"][i]
            labels.extend([label.get(ent) if ent in label else [0] for ent in all_possible_ent_pair])
        feature_set = torch.stack(feature_set)
        # labels = torch.Tensor(labels)
        return feature_set, labels

    def all_possible_pair(self, num_ent: int):
        return list(
            filter(lambda x: x[1] != x[0],
                   product(range(num_ent),
                           range(num_ent)))
        )

    def aggregate_entities(self, vertexSet_list: list,
                           embedded_doc_list: torch.Tensor) -> list:
        """
        :param vertexSet_list: list of mentions positions in a doc in shape (batch, max_entity, max_mentions, 2)
        :param embedded_doc_list: embedded doc in shape (batch, 512, 768)
        :return: The logsumexp pooling of mentions for each entity in shape(max_entity, 768)
        """
        batch = []
        for vertexSet, embedded_doc in zip(vertexSet_list, embedded_doc_list):
            # logsumexp = lambda x: torch.log(torch.sum(torch.exp(x), axis=0)).unsqueeze(0)
            mention_positions = [list(filter(lambda x: x[0] <= 512 and x[1] <= 512, ent)) for ent in vertexSet]
            mention_positions = [list(filter(lambda x: len(x) > 0, mnt)) for mnt in mention_positions]
            mention_positions = list(filter(lambda x: len(x) > 0, mention_positions))
            #  doc contains aggregated entity embedding, in shape(ent, 768)
            doc = [torch.logsumexp(torch.concat([embedded_doc[pos[0] + 1:pos[1] + 1] for pos in mnt]), dim=0)
                   for mnt in mention_positions]
            doc = torch.stack(doc)
            batch.append(doc)
        return batch