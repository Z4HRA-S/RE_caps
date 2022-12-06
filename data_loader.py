from torch.utils.data import Dataset
import torch
import transformers as ppb
import json
from typing import List, Dict
import traceback
from copy import deepcopy

max_ent = 42
max_lbl = 151


class DocRED(Dataset):
    def __init__(self, data_path):
        tokenizer_class, pretrained_weights = (ppb.BertTokenizer,
                                               'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights,
                                                         padding_side="right",
                                                         pad_token="[PAD]"
                                                         )
        self.data, self.rel2id = self.__read_data__(data_path)

    def __len__(self):
        return len(self.data)

    def __read_data__(self, data_dir: "str"):
        with open(data_dir, "r") as file:
            data = json.loads(file.read())
        data = [d for d in data if len(d["labels"]) > 0]
        rel2id_dir = "/".join(data_dir.split("/")[:-1] + ["rel2id.json"])
        with open(rel2id_dir, "r") as file:
            rel2id = json.loads(file.read())

        return data, rel2id

    def __join_sents__(self, doc: Dict):
        len_sents = [len(sent) for sent in doc["sents"]]
        for ent in doc["vertexSet"]:
            for mnt in ent:
                mnt["pos"][0] += sum(len_sents[:mnt["sent_id"]])
                mnt["pos"][1] += sum(len_sents[:mnt["sent_id"]])

        doc["sents"] = sum(doc["sents"], [])
        doc["sents"] = list(map(lambda x: x.lower(), doc["sents"]))
        return doc

    def __tokenize__(self, doc: List[str]):
        tokenized_doc = self.tokenizer.encode_plus(doc[:510],
                                                   add_special_tokens=True,
                                                   padding="max_length",
                                                   max_length=512,
                                                   return_tensors="pt")
        return tokenized_doc

    def __preprocess__(self, doc):
        # doc = self.__add_special_entity_token__(doc)
        doc = self.__join_sents__(doc)
        tokenized_doc = self.__tokenize__(doc["sents"])
        input_ids = torch.squeeze(tokenized_doc["input_ids"], 1)
        attention_mask = torch.squeeze(tokenized_doc["attention_mask"], 1)
        vertexSet = [[mnt["pos"] for mnt in ent] for ent in doc["vertexSet"]]
        labels = torch.zeros(len(vertexSet), len(vertexSet), 96)
        for triple in doc["labels"]:
            i, j, k = triple["h"], triple["t"], self.rel2id[triple["r"]]
            labels[i][j][k] = 1

        processed_doc = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "entity_list": vertexSet,
            "entity_num": len(doc["vertexSet"]),
            "label": labels
        }
        return processed_doc

    def custom_collate_fn(self, batch):
        batch_data = {
            "entity_list": [d["entity_list"] for d in batch],
            "input_ids": torch.concat([d["input_ids"] for d in batch]),
            "attention_mask": torch.concat([d["attention_mask"] for d in batch]),
            "entity_num": [d["entity_num"] for d in batch],
            "label": [d["label"] for d in batch]
        }
        return batch_data

    def __getitem__(self, idx):
        try:
            data = deepcopy(self.data[idx])
            item = self.__preprocess__(data)
            del data
            return item
        except Exception as e:
            print(self.data[idx])
            traceback.format_exc()

    def get_token_embedding(self):
        return len(self.tokenizer)
