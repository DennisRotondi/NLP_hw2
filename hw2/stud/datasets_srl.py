import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Any, Dict
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import numpy as np
class Dataset_SRL_34(Dataset):
    def __init__(self, sentences: Dict[str, List[str]], need_train: bool):
        # if the dataset is for a model that need_train we assume to have labels
        self.has_labels = need_train 
        if self.has_labels:
            # with this function we create a dict to encode and decode easily labels of roles
            self.create_labels_id_mapping_roles()
        self.data = self.make_data(sentences)

    def create_labels_id_mapping_roles(self):
        # these labels have been extracted studying the dataset from the notebook
        labels = ['agent', 'theme', 'beneficiary', 'patient', 'topic', 'goal', 'recipient', 
            'co-theme', 'result', 'stimulus', 'experiencer', 'destination', 'value', 'attribute', 
            'location', 'source', 'cause', 'co-agent', 'time', 'co-patient', 'product', 'purpose', 
            'instrument', 'extent', 'asset', 'material', '_']
        self.labels_to_id = {lab: i for i, lab in enumerate(labels)}
        self.id_to_labels = {i: lab for i, lab in enumerate(labels)}

    def make_data(self, sentences):
        data = list() 
        for j, ids in enumerate(sentences):
            # we extract the position of the predicates
            predicate_positions = [i for i, p in enumerate(sentences[ids]["predicates"]) if p != '_' and p != 0]
            sentence = sentences[ids]["lemmas"] 
            s_length = len(sentence)
            for predicate_position in predicate_positions:
                item = dict()
                item["id"] = j
                item["len"] = s_length

                # here I build a tuple to attain the same input as proposed by "Shi - Lin 19" after the embedding
                # I have also added the predicate disambiguation of that value to check if it's possible to improve the result.
                item["input"] = (sentence, [sentence[predicate_position],sentences[ids]["predicates"][predicate_position]])
                if self.has_labels:
                    # the desired output are the labels already encoded for each role associated to the sentence[predicate_position]
                    item["role_labels"] = torch.as_tensor([self.labels_to_id[i] for i in sentences[ids]["roles"][predicate_position]])
                    # note that input and output have different sizes after the input embedding if a word_piece tokenizer is used
                    # (as in my case), but don't worry, the model will produce an average mean of piece_tokens so that the sizes
                    # will be compatible.
                data.append(item)               
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# class Dataset_SRL_234(dataset_srl_34):
#     pass

# class Dataset_SRL_1234(dataset_srl_234):
#     pass

class SRL_DataModule(pl.LightningDataModule):
    def __init__(self, hparams: dict, task: str, sentences: Dict[str, List[str]], sentences_test: Dict[str, List[str]] = None) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.sentences = sentences
        self.sentences_test = sentences_test
        assert(task in ["1234", "234", "34"])
        self.task = task

    def setup(self) -> None:
        if self.task == "34":
            DATASET = Dataset_SRL_34
        # elif self.task == "234":
        #     DATASET = Dataset_SRL_234
        # else:
        #     DATASET = Dataset_SRL_1234
        self.data_train = DATASET(self.sentences, self.hparams.need_train)
        if self.sentences_test: 
            self.data_test = DATASET(self.sentences_test, self.hparams.need_train)
        # now we free up space
        delattr(self, "sentences")
        delattr(self, "sentences_test")

    def train_dataloader(self):
        return DataLoader(
                self.data_train, 
                batch_size = self.hparams.batch_size, 
                shuffle = self.hparams.need_train, # if we need train we shuffle it
                num_workers = self.hparams.n_cpu,
                collate_fn = self.collate_fn,
                persistent_workers = True
            )

    def val_dataloader(self):
       return DataLoader(
                self.data_test, 
                batch_size = self.hparams.batch_size, 
                shuffle = False,
                num_workers = self.hparams.n_cpu,
                collate_fn = self.collate_fn,
                persistent_workers = True
            )
    # here we define our collate function to apply the padding
    # ispiration from transformers notebook
    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.language_model_name)
        batch_out = tokenizer.batch_encode_plus(
            [sentence["input"] for sentence in batch],
            return_tensors="pt",
            padding=True,
            # We use this argument because the texts in our dataset are lists of words.
            is_split_into_words=True,
        )
        ids = list()
        word_ids = list()
        labels = list()
        lens = list()
        for i, sentence in enumerate(batch):
            ids.append(sentence["id"])
            lens.append(sentence["len"])

            w_id = np.array(batch_out.word_ids(batch_index=i), dtype=np.float64)
            # since w_id contains None values we want to remove them to have the conversion in tensor
            # we'll do so by creating a "special_token index" that is +1 higher than the last word token
            special_idx = np.nanmax(w_id) + 1
            w_id[np.isnan(w_id)] = special_idx
            # we need to mask vectors belonging to the "second" sentence (the one made to produce the embedding) 
            # they all will get a value different from all the other tokens so it will be easy to remove
            w_id[batch_out["token_type_ids"][i]] = special_idx
            word_ids.append(w_id)
            if self.hparams.need_train:
                labels.append(sentence["role_labels"])

        labels = pad_sequence(
                [torch.as_tensor(label) for label in labels],
                batch_first=True,
                padding_value=-100
            )
        batch_out["id"] = torch.as_tensor(ids)
        # np conversion of the list to speedup the tensor creation
        batch_out["word_id"] = torch.as_tensor(np.array(word_ids), dtype=torch.long) 
        batch_out["labels"] = torch.as_tensor(labels)
        return batch_out