import json
from os import device_encoding
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from model import Model
import dataclasses
from dataclasses import dataclass, asdict
from transformers import AutoModel
from transformers_embedder.embedder import TransformersEmbedder
import transformers_embedder as tre
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer
import spacy
from spacy.tokens import Doc
from utils import evaluate_predicate_disambiguation, evaluate_predicate_identification
from utils import evaluate_argument_classification, evaluate_argument_identification
import csv
import requests
import json
from tqdm import tqdm
import copy

# all "package relative imports" here, to avoid repeat code in the notebook as I did for hw1
try:
    from .datasets_srl import Dataset_SRL_34, Dataset_SRL_1234  # NOTE: relative import to work with docker
except:
    print("working with notebook need an 'absolute' import")
    from datasets_srl import Dataset_SRL_34, Dataset_SRL_1234


def build_model_34(language: str, device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(language=language, device=device, task="34")


def build_model_234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    raise NotImplementedError


def build_model_1234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    raise NotImplementedError

@dataclass
class HParams():
    # dataset stuff
    need_train: bool = True
    batch_size: int = 128 #128 for 1234, 256 for 34
    n_cpu: int = 8
    role_classes: int = 27 # number of different SRL roles for this homework
    pos_tag_tokens: int = 17
    # model stuff
    language_model_name: str = "bert-base-uncased" #"bert-base-multilingual-cased"
    lr: int = 1e-3
    wd: int = 0
    embedding_dim: int = 768
    hidden_dim: int = 400
    bidirectional: bool = True 
    num_layers: int = 1
    dropout: float = 0.3
    trainable_embeddings: bool = True 
    pos_tag_emb_dim: int = 232
    language: str = "EN"
    task: str = "34"

class StudentModel(Model):
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self, language: str, device: str, task: str):
        # load the specific model for the input language
        self.device = device
        hparams = HParams()
        hparams.need_train = False
        hparams.language = language
        hparams.task = task
        self.hparams = hparams
        # this has been a common problem between students, we need to init rapidly the student model
        # so for now we set the model to None and at prediction time we upload it!
        self.model = None 
    def predict(self, sentence):
        if self.model is None:
            if self.hparams.task == "34":
                self.model = SRL_34.load_from_checkpoint(f"model/SRL_34_{self.hparams.language}", strict=False).to(self.device)
        return self.model.predict(sentence)

class SRL_34(pl.LightningModule):
    def __init__(self, hparams: dict, sentences_for_evaluation=None) -> None:
        super(SRL_34, self).__init__()
        self.save_hyperparameters(hparams)
        self.transformer_model = AutoModel.from_pretrained(self.hparams.language_model_name, output_hidden_states=True)
        for param in self.transformer_model.parameters():
            param.requires_grad = False
        if self.hparams.trainable_embeddings:
            # I can unfreeze only some layers due to my limited gpu card memory.
            unfreeze = [10, 11]
            for i in unfreeze:
                for param in self.transformer_model.encoder.layer[i].parameters():
                    param.requires_grad = True
        
        if sentences_for_evaluation is not None:
            self.sentences_for_evaluation = sentences_for_evaluation

        self.lstm = nn.LSTM(self.hparams.embedding_dim, self.hparams.hidden_dim, 
                            bidirectional = self.hparams.bidirectional,
                            num_layers = self.hparams.num_layers, 
                            dropout = self.hparams.dropout if self.hparams.num_layers > 1 else 0,
                            batch_first = True,
                            )
        lstm_output_dim = self.hparams.hidden_dim if self.hparams.bidirectional is False else self.hparams.hidden_dim * 2
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, self.hparams.role_classes)
        # experiment to have a better training, we work only on the identification here
        self.identifier = nn.Linear(lstm_output_dim, 1)
        # the tokenizer here is useful to speedup the prediction process!
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.language_model_name)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_kwargs = {
          "input_ids": x["input_ids"], 
          "attention_mask": x["attention_mask"],
          "token_type_ids": x["token_type_ids"]
        }
        # if reduction sum for example
        transformers_outputs = self.transformer_model(**model_kwargs)
        transformers_outputs_sum = torch.stack(transformers_outputs.hidden_states[-4:], dim=0).sum(dim=0)
        # I use the Riccardo Orlando merger to average the tokens of the initial word splitted by the tokenizer
        embeddings = TransformersEmbedder.merge_scatter(transformers_outputs_sum, x["word_id"])[:,:-1,:]
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        predict_a = self.classifier(o)
        identification = torch.sigmoid(self.identifier(o))
        return {"class": predict_a, "id": identification}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=self.hparams.wd)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'loss',
                "frequency": 1
            },
        }

    def loss_function(self, predictions, labels):
        identification = predictions["id"]
        predictions = predictions["class"]
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1)
        CE = F.cross_entropy(predictions, labels, ignore_index = -100)

        mask = labels != -100
        identification = identification.view(-1)
        predicate_labels = (labels[mask] != self.hparams.role_classes-1).float() #26 is the label of "_"
        BCE = F.binary_cross_entropy(torch.sigmoid(identification[mask]), predicate_labels)
        return {"loss": CE+BCE, "CE": CE, "BCE": BCE}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> Dict[str, torch.Tensor]:
        output = self(batch)
        loss = self.loss_function(output, batch["labels"])
        self.log_dict(loss)
        return loss['loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        output = self(batch)
        loss = self.loss_function(output, batch["labels"])
        return {"loss_val": loss['loss']}

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        avg_loss = torch.stack([x["loss_val"] for x in outputs]).mean()
        predict = self.predict(self.sentences_for_evaluation, require_ids=True)
        eac = evaluate_argument_classification(self.sentences_for_evaluation, predict)
        eai = evaluate_argument_identification(self.sentences_for_evaluation, predict)
        dict_ai = dict()
        dict_ac = dict()
        for key in eai:
            dict_ai[key+"_ai"] = float(eai[key])
            dict_ac[key] = float(eac[key])
        self.log_dict(dict_ai)
        self.log_dict(dict_ac)
        self.log_dict({"avg_val_loss": avg_loss})
        return {"avg_val_loss": avg_loss}

    def predict(self, sentences: Dict[str, List[str]], require_ids = False):
        """
            INPUT:
            - sentence:
                {
                    "words":
                        [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                    "lemmas":
                        ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                    "predicates":
                        ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                }
            - require_ids:
                is a parameter to keep track of the sentence id if set to true we have a corresponce between input output (useful if
                we are working at training time to exploit the utils functions of this homework.)
            OUTPUT:
                {
                "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence.
                }
                or a dict of them with key the id of the sentence if require_ids is True.
        """
        # even if with the docker we have a sentence at a time, I decided to do a "batch" approach to be able to compute all the metrics
        # at training time easily exploiting the utils functions of this homework.
        # those two functions allows me to encapsulate the prediction functions
        def encode_sentence(self, sentence: List[str], predicate_position: int):
            # this is in brief what we do in the training time, since we are working with
            # a sentence at a time this is the best way to proceed I've thought about.
            input = (sentence["lemmas"], [sentence["lemmas"][predicate_position],sentence["predicates"][predicate_position]])
            # print(input)
            batch_out = self.tokenizer.batch_encode_plus(
                    [input],
                    return_tensors="pt",
                    is_split_into_words=True,
                )
            w_id = np.array(batch_out.word_ids(0), dtype=np.float64)
            special_idx = np.nanmax(w_id) + 1
            w_id[np.isnan(w_id)] = special_idx
            w_id[batch_out["token_type_ids"][0]] = special_idx
            batch_out["word_id"] = torch.as_tensor(np.array([w_id]), dtype=torch.long) 
            return batch_out
        
        def predict_roles(self, sentence: List[str]):
            roles = dict()
            predicate_positions = [i for i, p in enumerate(sentence["predicates"]) if p != '_']
            for ppos in predicate_positions:
                input = encode_sentence(self, sentence, ppos).to(self.device)
                output = self(input)["class"]
                output = torch.argmax(output,-1)[0].tolist()
                roles[ppos] = [self.id_to_labels[id] for id in output]
            return {"roles": roles}

        self.eval()
        if not hasattr(self, 'id_to_labels'):
            _, self.id_to_labels = Dataset_SRL_34.create_labels_id_mapping_roles()
        with torch.no_grad():
            if not require_ids:
                return predict_roles(self, sentences)
            predictions = dict()
            for id in sentences:
                predictions[id] = predict_roles(self, sentences[id])
            return predictions

class SRL_1234(pl.LightningModule):
    def __init__(self, hparams: dict, sentences_for_evaluation=None) -> None:
        super(SRL_1234, self).__init__()
        self.save_hyperparameters(hparams)
        if sentences_for_evaluation is not None:
            self.sentences_for_evaluation = sentences_for_evaluation
        self.transformer_model = TransformersEmbedder(
                                 self.hparams.language_model_name,
                                 subword_pooling_strategy="sparse", 
                                 layer_pooling_strategy="mean",
                                 fine_tune = False,
                                )
        if self.hparams.trainable_embeddings:
            unfreeze = [10, 11]
            for i in unfreeze:
                for param in self.transformer_model.transformer_model.encoder.layer[i].parameters():
                    param.requires_grad = True
        # we will use ad padding idx the highest id
        n_pt = self.hparams.pos_tag_tokens+1
        pt_emb_dim = self.hparams.pos_tag_emb_dim
        self.pt_embed = nn.Embedding(n_pt, pt_emb_dim, padding_idx=self.hparams.pos_tag_tokens)
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.gru = nn.GRU(self.hparams.embedding_dim+pt_emb_dim, self.hparams.hidden_dim, 
                            bidirectional=self.hparams.bidirectional,
                            num_layers=self.hparams.num_layers, 
                            dropout = self.hparams.dropout if self.hparams.num_layers > 1 else 0,
                            batch_first=True)
        gru_output_dim = self.hparams.hidden_dim if self.hparams.bidirectional is False else self.hparams.hidden_dim * 2
        self.classifier = nn.Linear(gru_output_dim, 1)
        # the tokenizer here is useful to speedup the prediction process!
        self.tokenizer = tre.Tokenizer(self.hparams.language_model_name)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        transformers_outputs = self.transformer_model(**input)
        w_embeddings = transformers_outputs.word_embeddings[:,1:-1,:] 
        pt_embeddings = self.pt_embed(input["pos_tags"])
        embeddings = torch.cat((w_embeddings, pt_embeddings), dim=-1)
        # embeddings = w_embeddings + pt_embeddings another experiment
        # de = self.dropout(embeddings)
        o2, _ = self.gru(embeddings)
        return self.sigmoid(self.classifier(o2))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=self.hparams.wd)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=1e-8)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'loss',
                "frequency": 1
            },
        }

    def loss_function(self, predictions, labels, ignore_index = -100):
        predictions = predictions.view(-1)
        labels = labels.view(-1)
        mask = labels != ignore_index
        # print(predictions[mask].shape)
        # print(labels[mask].float().shape)
        BCE = F.binary_cross_entropy(predictions[mask], labels[mask].float())
        return {"loss": BCE}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> Dict[str, torch.Tensor]:
        output = self(batch)
        loss = self.loss_function(output, batch["labels"])
        self.log_dict(loss)
        return loss['loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        output = self(batch)
        loss = self.loss_function(output, batch["labels"])
        return {"loss_val": loss['loss']}

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        avg_loss = torch.stack([x["loss_val"] for x in outputs]).mean()
        predict = self.predict(self.sentences_for_evaluation, require_ids=True, training=True)
        pi = evaluate_predicate_identification(self.sentences_for_evaluation, predict)
        pi_d = dict()
        for key in pi:
            # we need to convert to float to compute plots faster
            pi_d[key] = float(pi[key])
        self.log_dict(pi_d)
        self.log_dict({"avg_val_loss": avg_loss})
        return {"avg_val_loss": avg_loss}

    def predict(self, sentences: Dict[str, List[str]], require_ids = False, training = False):
        """
            INPUT:
            - sentence:
                {
                    "words":
                        [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                    "lemmas":
                        ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                }
            - require_ids:
                is a parameter to keep track of the sentence id if set to true we have a corresponce between input output (useful if
                we are working at training time to exploit the utils functions of this homework.)
            OUTPUT:
                {
                "predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
                "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence.
                }
                or a dict of them with key the id of the sentence if require_ids is True.
                if training is set to True it will return a 0-1 encoding to indicate the predicate
        """
        # even if with the docker we have a sentence at a time, I decided to do a "batch" approach to be able to compute all the metrics
        # at training time easily exploiting the utils functions of this homework.
        # those two functions allows me to encapsulate the prediction functions
        def encode_sentence(self, sentence: List[str]):
            # this is in brief what we do in the training time, since we are working with
            # a sentence at a time this is the best way to proceed I've thought about.
            input = sentence["words"]
            # print(input)
            batch_out = self.tokenizer(
                        [input],
                        return_tensors = True,
                        is_split_into_words=True
                    )        
            ###
            # NOTE: there is at least a sentence 2003/a/58/562_24:1 with a '' token, I think this is a bug but since
            # I'm not allowed to change the dataset I have to manually replace it with a " "
            try:
                sentence = Doc(self.nlp.vocab, input)
            except:
                # we have to fix '' bug
                input = [w if w != '' else " " for w in input]
                sentence = Doc(self.nlp.vocab, input)
            doc = self.nlp(sentence)
            pos_tags = list()
            for token in doc:
                pos_tags.append(self.tags_to_id[token.pos_] if token.pos_ != "SPACE" else self.tags_to_id["PUNCT"])
            batch_out["pos_tags"] = torch.as_tensor([pos_tags])
            return batch_out
        
        def predict_roles(self, sentence: List[str], training = False):
            predict = dict()
            input = encode_sentence(self, sentence).to(self.device)
            output = self(input)
            predict["predicates"] = torch.round(output).view(-1).int().tolist()
            if training:
                predict["predicates"] = ["_" if p == 0 else "1" for p in predict["predicates"]]
            return predict

        self.eval()
        if not hasattr(self, 'nlp'):
            taggers = {"EN":"en_core_web_sm", "ES":"es_core_news_sm", "FR":"fr_core_news_sm"}
            self.nlp = spacy.load(taggers[self.hparams.language])
            self.tags_to_id, _ = Dataset_SRL_1234.create_ptag_id_mapping()
        with torch.no_grad():
            if not require_ids:
                return predict_roles(self, sentences, training)
            predictions = dict()
            for id in sentences:
                predictions[id] = predict_roles(self, sentences[id], training)
            return predictions

class SRL_234():
    def __init__(self, hparams: dict) -> None:
        # ok, there are many possible ways to solve this task, the most efficient are very
        # similar to the other 2 I've implemented, so I've decided to work on this extra to do the simplest
        # thing possible: I let AMuSE-WSD do the work. Given the predicted sense is possible to encode this
        # information to use a neural network or any machine learning model properly fitted. 
        def read_in_dict(in_file: str):
            with open(in_file) as file:
                tsv_file = csv.reader(file, delimiter="\t")
                dict = {}
                for line in tsv_file:
                    key, val = line[0], line[1]
                    dict[key] = val
            return dict
        bn2va_file = "../../model/VA_bn2va.tsv"
        va_info_file = "../../model/VA_frame_info.tsv"
        # convert from bn synset to verbatlas frame
        self.bn2va = read_in_dict(bn2va_file)
        # retrive the sense from verbatlas frame
        self.va_info = read_in_dict(va_info_file)
        self.url = 'http://nlp.uniroma1.it/amuse-wsd/api/model'
        self.headers = {'accept': 'application/json', 'Content-Type': 'application/json'}
        self.language = hparams["language"]
        if not hparams["need_train"]:
            self.layer = torch.load("model/SRL_234_layer2.ckpt")
        self.hparams = hparams

    def use_AMuSE_WSD(self, sentence: Dict[str, List[str]]):
        # http://nlp.uniroma1.it/amuse-wsd/api-documentation
        stringa = " ".join(sentence["lemmas"])
        dict =  [{"text": stringa, "lang" : self.language}]
        payload = json.dumps(dict)
        r = requests.post( self.url, data=payload, headers=self.headers)
        sol = r.json()[0]['tokens']
        senses = list()
        for i, n in enumerate(sentence["predicates"]):
            if n == 0:
                senses.append("_")
            else:
                bn_id = sol[i]["bnSynsetId"]
                if bn_id not in self.bn2va:
                    # if we have a "noun prediction" it's clearly not a verb
                    senses.append("_")
                    continue
                vf_from_bn = self.bn2va[bn_id]
                senses.append(self.use_filtered_predictions(vf_from_bn))
        return {"predicates" : senses}

    def use_filtered_predictions(self,vf_from_bn):
        if hasattr(self, 'filter') and self.va_info[vf_from_bn] in self.filter:
            return self.filter[self.va_info[vf_from_bn]]
        else:
            return self.va_info[vf_from_bn]

    def predict(self, sentences: Dict[str, List[str]], require_ids = False, simple = True):
        """
        INPUT:
        {
            "words": [...], # SAME AS BEFORE
            "lemmas": [...], # SAME AS BEFORE
            "predicates":
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
        }
        OUTPUT:
        {
            "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
            "roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence.
        }
        """
        if not require_ids:
            amuse_pred = self.use_AMuSE_WSD(sentences)
            sentences["predicates"] = amuse_pred
            if not hasattr(self, 'srl34'):
                self.srl34 = SRL_34.load_from_checkpoint(f"model/SRL_34_{self.hparams.language}", strict=False).to(self.device)
            return self.srl34.predict(sentences)
        # ELSE    
        predictions = dict()
        # need a copy to avoid side effects
        sentences2 = copy.deepcopy(sentences)
        for id in tqdm(sentences2):
            # NOTE: tqdm not in the other predicts since it breaks the training prints
            sentences2[id]["predicates"] = [0 if i == '_' else 1 for i in sentences2[id]["predicates"]]
            # here as always "private" evaluation
            predictions[id] = self.use_AMuSE_WSD(sentences2[id])
        return predictions