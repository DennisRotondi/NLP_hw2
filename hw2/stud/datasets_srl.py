import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List, Any, Dict

class Dataset_SRL_34(Dataset):


    
    pass

# class Dataset_SRL_234(dataset_srl_34):
#     pass

# class Dataset_SRL_1234(dataset_srl_234):
#     pass

class SRL_DataModule(pl.LightningDataModule):
    def __init__(self, hparams: object, task: str, sentences: Dict[str, List[str]], sentences_test: Dict[str, List[str]]=None) -> None:
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
        self.data_train = DATASET(self.sentences)
        if self.sentences_test: 
            self.data_test = DATASET(self.sentences_test)
    #NEED TO ADD COLLATEfn
    def train_dataloader(self):
        return DataLoader(
                self.data_train, 
                batch_size = self.hparams.batch_size, 
                shuffle = True,
                num_workers = self.hparams.n_cpu,
                pin_memory = True,
                collate_fn = self.collate_fn,
                persistent_workers = True
            )

    def val_dataloader(self):
       return DataLoader(
                self.data_test, 
                batch_size = self.hparams.batch_size, 
                shuffle = False,
                num_workers = self.hparams.n_cpu,
                pin_memory = True,
                collate_fn = self.collate_fn,
                persistent_workers = True
            )
    # here we define our collate function
    # from transform notebook
    def collate_fn(batch) -> Dict[str, torch.Tensor]:
        batch_out = tokenizer(
            [sentence["tokens"] for sentence in batch],
            return_tensors="pt",
            padding=True,
            # We use this argument because the texts in our dataset are lists of words.
            is_split_into_words=True,
        )
        labels = []
        ner_tags = [sentence["ner_tags"] for sentence in batch]
        for i, label in enumerate(ner_tags):
        # obtains the word_ids of the i-th sentence
        word_ids = batch_out.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
            label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
            label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to -100 so they are automatically
            # ignored in the loss function.
            else:
            label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
        
        # pad the labels with -100
        batch_max_length = len(max(labels, key=len))
        labels = [l + ([-100] * abs(batch_max_length - len(l))) for l in labels]
        batch_out["labels"] = torch.as_tensor(labels)
        return batch_out