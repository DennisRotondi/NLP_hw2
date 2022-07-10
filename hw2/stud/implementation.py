import json
import random
import numpy as np
from typing import List, Tuple, Dict
from model import Model
import dataclasses
from dataclasses import dataclass, asdict
from transformers import AutoModel
from transformers_embedder.embedder import TransformersEmbedder
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl


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
    return Baseline(language=language)


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
    batch_size: int = 3
    n_cpu: int = 8
    # model stuff
    language_model_name: str = "bert-base-uncased"
    lr: int = 1e-3
    wd: int = 1e-5
    embedding_dim: int = 768
    hidden_dim: int = 512
    bidirectional: bool = True 
    num_layers: int = 5
    dropout: float = 0.4
    trainable_embeddings: bool = True 
    role_classes: int = 27 # number of different SRL roles for this homework

class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, language: str, return_predicates=False):
        self.language = language
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        print(sentence)
        for pos in sentence["pos_tags"]:
            prob = self.baselines["predicate_identification"].get(pos, dict()).get(
                "positive", 0
            ) / self.baselines["predicate_identification"].get(pos, dict()).get(
                "total", 1
            )
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)

        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(
            zip(sentence["lemmas"], predicate_identification)
        ):
            if (
                not is_predicate
                or lemma not in self.baselines["predicate_disambiguation"]
            ):
                predicate_disambiguation.append("_")
            else:
                predicate_disambiguation.append(
                    self.baselines["predicate_disambiguation"][lemma]
                )
                predicate_indices.append(idx)

        argument_identification = []
        for dependency_relation in sentence["dependency_relations"]:
            prob = self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get("positive", 0) / self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get(
                "total", 1
            )
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(
            sentence["dependency_relations"], argument_identification
        ):
            if not is_argument:
                argument_classification.append("_")
            else:
                argument_classification.append(
                    self.baselines["argument_classification"][dependency_relation]
                )

        if self.return_predicates:
            return {
                "predicates": predicate_disambiguation,
                "roles": {i: argument_classification for i in predicate_indices},
            }
        else:
            return {"roles": {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path="data/baselines.json"):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self, language: str):
        # load the specific model for the input language
        self.language = language

    def predict(self, sentence):
        """
        --> !!! STUDENT: implement here your predict function !!! <--

        Args:
            sentence: a dictionary that represents an input sentence, for example:
                - If you are doing argument identification + argument classification:
                    {
                        "words":
                            [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                        "lemmas":
                            ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                        "predicates":
                            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                    },
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "predicates":
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    },
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        # NOTE: you do NOT have a "predicates" field here.
                    },

        Returns:
            A dictionary with your predictions:
                - If you are doing argument identification + argument classification:
                    {
                        "roles": list of lists, # A list of roles for each predicate in the sentence.
                    }
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
                        "roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence.
                    }
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
                        "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence.
                    }
        """
        pass


class SRL_34(pl.LightningModule):
    def __init__(self, hparams: dict) -> None:
        super(SRL_34, self).__init__()
        self.save_hyperparameters(hparams)
        self.transformer_model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        if not self.hparams.trainable_embeddings:
            for param in self.transformer_model.parameters():
                param.requires_grad = False
        self.lstm = nn.LSTM(self.hparams.embedding_dim, self.hparams.hidden_dim, 
                            bidirectional = self.hparams.bidirectional,
                            num_layers = self.hparams.num_layers, 
                            dropout = self.hparams.dropout if self.hparams.num_layers > 1 else 0,
                            batch_first = True,
                            )
        lstm_output_dim = self.hparams.hidden_dim if self.hparams.bidirectional is False else self.hparams.hidden_dim * 2
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, self.hparams.role_classes)
        ##### ---- #### ---- #### ---- ####

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
        return self.classifier(o)

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

    def loss_function(self, input, target):
        CE = F.CrossEntropyLoss(input, target, ignore_index = -100)
        return {"loss": CE}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx) -> Dict[str, torch.Tensor]:
        output = self(batch)
        loss = self.loss_function(output, batch["labels"])
        self.log_dict(loss)
        return loss['loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        output = self(batch)
        loss = self.loss_function(output, batch["labels"])
        return {"loss_vae_val": loss['loss']}

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        avg_loss = torch.stack([x["loss_vae_val"] for x in outputs]).mean()
        self.log_dict({"avg_val_loss_vae": avg_loss})
        return {"avg_val_loss_vae": avg_loss}