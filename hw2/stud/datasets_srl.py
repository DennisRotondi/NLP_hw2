import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Any, Dict, Optional
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
import transformers_embedder as tre
import numpy as np
import spacy
from spacy.tokens import Doc

class Dataset_SRL_34(Dataset):
    def __init__(self, sentences: Dict[str, List[str]], need_train: bool, language: str):
        # if the dataset is for a model that need_train we assume to have labels
        self.has_labels = need_train 
        self.language = language
        if self.has_labels:
            # with this function we create a dict to encode and decode easily labels of roles
            self.labels_to_id, self.id_to_labels = Dataset_SRL_34.create_labels_id_mapping_roles()
        self.data = self.make_data(sentences)

    @staticmethod
    def create_labels_id_mapping_roles():
        # these labels have been extracted studying the dataset from the notebook
        labels = ['agent', 'theme', 'beneficiary', 'patient', 'topic', 'goal', 'recipient', 
            'co-theme', 'result', 'stimulus', 'experiencer', 'destination', 'value', 'attribute', 
            'location', 'source', 'cause', 'co-agent', 'time', 'co-patient', 'product', 'purpose', 
            'instrument', 'extent', 'asset', 'material', '_']
        return {lab: i for i, lab in enumerate(labels)}, {i: lab for i, lab in enumerate(labels)}

    def make_data(self, sentences):
        data = list() 
        for ids in sentences:
            # we extract the position of the predicates
            predicate_positions = [i for i, p in enumerate(sentences[ids]["predicates"]) if p != '_' and p != 0]
            sentence_l = sentences[ids]["lemmas"] 
            for predicate_position in predicate_positions:
                item = dict()
                # here I build a tuple to attain the same input as proposed by "Shi - Lin 19" after the embedding
                # I have also added the predicate disambiguation of that value to check if it's possible to improve the result.
                item["input"] = (sentence_l, [sentence_l[predicate_position], sentences[ids]["predicates"][predicate_position]])
                if self.has_labels:
                    # the desired output are the labels already encoded for each role associated to the sentence[predicate_position]
                    item["role_labels"] = [self.labels_to_id[i] for i in sentences[ids]["roles"][predicate_position]]
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

class Dataset_SRL_1234(Dataset):
    def __init__(self, sentences: Dict[str, List[str]], need_train: bool, language: str):
        # if the dataset is for a model that need_train we assume to have labels
        self.has_labels = need_train 
        self.language = language
        self.tags_to_id, self.id_to_tags= Dataset_SRL_1234.create_ptag_id_mapping()
        self.data = self.make_data(sentences)

    @staticmethod
    def create_ptag_id_mapping():
        # these labels have been extracted studying the dataset from the notebook
        ptags = ['NOUN', 'ADV', 'VERB', 'SCONJ', 'DET', 'ADJ', 'ADP', 'PUNCT', 'PROPN', 'PART', 
                'NUM', 'CCONJ', 'AUX', 'PRON', 'SYM', 'X', 'INTJ']
        return {lab: i for i, lab in enumerate(ptags)}, {i: lab for i, lab in enumerate(ptags)}

    def make_data(self, sentences):
        data = list() 
        taggers = {"EN":"en_core_web_sm", "ES":"es_core_news_sm", "FR":"fr_core_news_sm"}
        nlp = spacy.load(taggers[self.language])
        for ids in sentences:
            item = dict()
            ###
            # NOTE: there is at least a sentence 2003/a/58/562_24:1 with a '' token, I think this is a bug but since
            # I'm not allowed to change the dataset I have to manually replace it with a " "
            sentence_w = sentences[ids]["words"]
            try:
                sentence = Doc(nlp.vocab, sentence_w)
            except:
                sentence_w = [w if w != '' else " " for w in sentence_w]
                sentence = Doc(nlp.vocab, sentence_w)
            doc = nlp(sentence)
            pos_tag = list()
            for token in doc:
                # we do not have spaces in our tags in normal cases, it could be a missclassification
                pos_tag.append(self.tags_to_id[token.pos_] if token.pos_ != "SPACE" else self.tags_to_id["PUNCT"])
            assert(len(sentence_w) == len(pos_tag))
            item["input"] = sentence_w
            item['pos_tags'] = pos_tag
            if self.has_labels:
                item["labels"] = [0 if i == '_' else 1 for i in sentences[ids]["predicates"]]       
            data.append(item)                      
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class SRL_DataModule(pl.LightningDataModule):
    def __init__(self, hparams: dict, task: str, language: str, sentences: Dict[str, List[str]], sentences_test: Dict[str, List[str]] = None) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.sentences = sentences
        self.sentences_test = sentences_test
        assert(task in ["1234", "234", "34"])
        self.task = task
        self.language = language
        self.collates = {"34": self.collate_fn_34, "1234": self.collate_fn_1234}

    def setup(self, stage: Optional[str] = None) -> None:
        
        if self.task == "34":
            self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.language_model_name)
            DATASET = Dataset_SRL_34
        # elif self.task == "234":
        #     DATASET = Dataset_SRL_234
        else:
            self.tokenizer = tre.Tokenizer(self.hparams.language_model_name)
            DATASET = Dataset_SRL_1234
        self.data_train = DATASET(self.sentences, self.hparams.need_train, self.language)

        if self.sentences_test: 
            self.data_test = DATASET(self.sentences_test, self.hparams.need_train, self.language)

    def train_dataloader(self):
        # change collate based on the task
        return DataLoader(
                self.data_train, 
                batch_size = self.hparams.batch_size, 
                shuffle = self.hparams.need_train, # if we need train we shuffle it
                num_workers = self.hparams.n_cpu,
                collate_fn = self.collates[self.task],
                pin_memory = True,
                persistent_workers = True
            )

    def val_dataloader(self):
        # change collate based on the task
        return DataLoader(
                    self.data_test, 
                    batch_size = self.hparams.batch_size, 
                    shuffle = False,
                    num_workers = self.hparams.n_cpu,
                    collate_fn = self.collates[self.task],
                    pin_memory = True,
                    persistent_workers = True
                )
    
    # here we define our collate function to apply the padding
    def collate_fn_34(self, batch) -> Dict[str, torch.Tensor]:
        batch_out = self.tokenizer.batch_encode_plus(
            [sentence["input"] for sentence in batch],
            return_tensors="pt",
            padding=True,
            # We use this argument because the texts in our dataset are lists of words.
            is_split_into_words=True,
        )
        word_ids = list()
        labels = list()
        for i, sentence in enumerate(batch):
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
        if self.hparams.need_train:
            labels = pad_sequence(
                    [torch.as_tensor(label) for label in labels],
                    batch_first=True,
                    padding_value=-100
                )
            batch_out["labels"] = torch.as_tensor(labels)
        # np conversion of the list to speedup the tensor creation
        batch_out["word_id"] = torch.as_tensor(np.array(word_ids), dtype=torch.long) 
        return batch_out

    def collate_fn_1234(self, batch) -> Dict[str, torch.Tensor]:
        batch_out = self.tokenizer(
            [sentence["input"] for sentence in batch],
            return_tensors = True,
            padding = True,
            is_split_into_words=True
        )
        batch_out["pos_tags"] = pad_sequence(
                                    [torch.as_tensor(item["pos_tags"]) for item in batch],
                                    batch_first=True,
                                    padding_value=self.hparams.pos_tag_tokens
                                )
        if self.hparams.need_train:
            batch_out["labels"] = pad_sequence(
                                    [torch.as_tensor(item["labels"]) for item in batch],
                                    batch_first=True,
                                    padding_value=-100
                                )
        return batch_out