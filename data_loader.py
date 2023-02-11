from torch.utils.data import Dataset
import torch
import transformers as ppb
import json
from typing import List, Dict
import traceback
from copy import deepcopy

types_id = {'LOC':0, 'PER': 1, 'TIME': 2, 'MISC': 3, 'ORG': 4, 'NUM': 5}

class DocRED(Dataset):
    def __init__(self, data_path, use_negative=False):
        self.use_negative = use_negative
        tokenizer_class, pretrained_weights = (ppb.BertTokenizer,
                                               'bert-base-uncased')
        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights,
                                                         padding_side="right",
                                                         pad_token="[PAD]"
                                                         )
        special_tokens_dict = {'additional_special_tokens': ['<e>', '</e>']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.data, self.rel2id = self.__read_data__(data_path)

    def __len__(self):
        return len(self.data)

    def __read_data__(self, data_dir: "str"):
        with open(data_dir, "r") as file:
            data = json.loads(file.read())
        if not self.use_negative:
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

    def __add_special_entity_token__(self, doc: Dict):
        """
        Add special token at the start and the ending of the entity mentions
        and fix the positions
        :return: Fixed position data
        """
        # shift for every entity in a sentence as we proceed to add special tokens
        # one by one in a sentence
        entities = {key: [] for key in range(len(doc["sents"]))}
        for ent_idx, entity in enumerate(doc["vertexSet"]):
            for mnt_idx, mention in enumerate(entity):
                entities[mention["sent_id"]].append({"mention": mention,
                                                     "ent_idx": ent_idx,
                                                     "mnt_idx": mnt_idx})
        for sent_id, mention_list in entities.items():
            shift = 0
            mention_list = sorted(mention_list, key=lambda x: x["mention"]["pos"][0])
            for mnt_dic in mention_list:
                mention = mnt_dic["mention"]
                ent_idx = mnt_dic["ent_idx"]
                mnt_idx = mnt_dic["mnt_idx"]
                start, end = mention["pos"]

                start += shift
                end += shift

                sent = doc["sents"][sent_id]
                sent = list(sent[:start]) + ["<e>"] + list(sent[start:end]) + ["</e>"] + list(sent[end:])

                doc["sents"][sent_id] = sent
                doc["vertexSet"][ent_idx][mnt_idx]["pos"] = [start, end + 2]
                shift += 2

        return doc

    def __tokenize__(self, doc: List[str]):
        tokenized_doc = self.tokenizer.encode_plus(doc[:510],
                                                   add_special_tokens=True,
                                                   padding="max_length",
                                                   max_length=512,
                                                   return_tensors="pt")
        return tokenized_doc

    def __preprocess__(self, doc):
        doc = self.__add_special_entity_token__(doc)
        doc = self.__join_sents__(doc)
        tokenized_doc = self.__tokenize__(doc["sents"])
        input_ids = torch.squeeze(tokenized_doc["input_ids"], 1)
        attention_mask = torch.squeeze(tokenized_doc["attention_mask"], 1)
        vertexSet = [[mnt["pos"] for mnt in ent] for ent in doc["vertexSet"]]
        types = [types_id[ent[0]["type"]] for ent in doc["vertexSet"]]

        labels = torch.zeros(len(vertexSet), len(vertexSet), 96)

        for triple in doc["labels"]:
            i, j, k = triple["h"], triple["t"], self.rel2id[triple["r"]]
            k = max(0, k - 1)
            labels[i][j][k] = 1

        processed_doc = {
            "types": types,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "entity_list": vertexSet,
            "entity_num": len(doc["vertexSet"]),
            "label": labels
        }
        return processed_doc

    def custom_collate_fn(self, batch):
        batch_data = {
            "types": [d["types"] for d in batch],
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


