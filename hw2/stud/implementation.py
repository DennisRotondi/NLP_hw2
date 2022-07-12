import json
from os import device_encoding
import random
import numpy as np
from typing import List, Tuple, Dict, Optional
from model import Model
from tqdm import tqdm
import dataclasses
from dataclasses import dataclass, asdict
from transformers import AutoModel
from transformers_embedder.embedder import TransformersEmbedder
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoTokenizer
# all "package relative imports" here, to avoid repeat code in the notebook as I did for hw1
try:
    from .datasets_srl import Dataset_SRL_34  # NOTE: relative import to work with docker
except:
    print("working with notebook need an 'absolute' import")
    from datasets_srl import Dataset_SRL_34


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
    batch_size: int = 256
    n_cpu: int = 8
    # model stuff
    language_model_name: str = "bert-base-uncased" #"bert-base-multilingual-cased"
    lr: int = 1e-3
    wd: int = 0
    embedding_dim: int = 768
    hidden_dim: int = 512
    bidirectional: bool = True 
    num_layers: int = 1
    dropout: float = 0.3
    trainable_embeddings: bool = False 
    role_classes: int = 27 # number of different SRL roles for this homework
    srl_34_ckpt: str = "model/srl_34_EN.ckpt"

class StudentModel(Model):
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self, language: str, device: str, task: str):
        # load the specific model for the input language
        self.language = language
        self.device = device
        self.task = task
        hparams = HParams()
        hparams.need_train = False
        self.hparams = hparams
        # this has been a common problem between students, we need to init rapidly the student model
        # so for now we set the model to None and at prediction time we upload it!
        self.model = None 
    def predict(self, sentence):
        if self.model is None:
            if self.task == "34":
                self.model = SRL_34.load_from_checkpoint(self.hparams.srl_34_ckpt).to(self.device)
        return self.model.predict(sentence)
        """
        --> !!! STUDENT: implement here your predict function !!! <--

        Args:
            sentence: a dictionary that represents an input sentence, for example:
                - If you are doing argument identification + argument classification:
                   ,
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

class SRL_34(pl.LightningModule):
    def __init__(self, hparams: dict, sentences_for_evaluation=None) -> None:
        super(SRL_34, self).__init__()
        self.save_hyperparameters(hparams)
        self.transformer_model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        if not self.hparams.trainable_embeddings:
            for param in self.transformer_model.parameters():
                param.requires_grad = False
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

    def loss_function(self, predictions, labels):
        predictions = predictions.view(-1, predictions.shape[-1])
        labels = labels.view(-1)
        CE = F.cross_entropy(predictions, labels, ignore_index = -100)
        # MSE = F.mse_loss
        return {"loss": CE}

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
        # predict = self.predict(self.sentences_for_evaluation, require_ids=True)
        # eac = evaluate_argument_classification(self.sentences_for_evaluation, predict)
        # eai = evaluate_argument_identification(self.sentences_for_evaluation, predict)
        # dict_ai = dict()
        # dict_ac = dict()
        # for key in eai:
        #     dict_ai[key+"_ai"] = float(eai[key])
        #     dict_ac[key] = float(eac[key])
        # self.log_dict(dict_ai)
        # self.log_dict(dict_ac)
        self.log_dict({"avg_val_loss": avg_loss})
        return {"avg_val_loss": avg_loss}

    def predict(self, sentences, require_ids = False):
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
                output = self(input)
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
            for id in tqdm(sentences):
                predictions[id] = predict_roles(self, sentences[id])
            return predictions


