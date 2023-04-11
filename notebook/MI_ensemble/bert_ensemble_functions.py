# -*- coding: utf-8 -*-
# ###### Load Libraries ######

import os
import re
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import transformers
import datasets
import huggingface_hub
import pyarrow
import torch
torch.backends.cuda.matmul.allow_tf32 = True
from datasets import load_dataset, load_metric, ClassLabel
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.examples.pbt_transformers.utils import (
    download_data,
    build_compute_metrics_fn,
)
from ray.tune.schedulers import PopulationBasedTraining
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from typing import TypeVar, Type
from numba import cuda

# ##### Functions ######

def preprocessing(dataset: DatasetDict,
                  text_column: str,
                  label_column: str,
                  id_column: str,
                  model_name: str,
                  train_proportion: float,
                  seed: int,
                  custom_tokenizer_dir: str = "my_result"
                  ) -> tuple:
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Values in `dataset` should be of type `DatasetDict` but got type '{type(dataset)}'")
    
    # Select columns to use
    print("Removing rows with missing value...")
    cols_to_remove = list(dataset['train'].features.keys())
    cols_to_remove.remove(id_column)
    cols_to_remove.remove(text_column)
    cols_to_remove.remove(label_column)
    dataset = dataset.remove_columns(cols_to_remove)
    if 'text' not in dataset['train'].features.keys():
        dataset = dataset.rename_column(text_column, "text")
    if label_column not in dataset['train'].features.keys():
        dataset = dataset.rename_column(label_column, "label")
    if id_column not in dataset['train'].features.keys():
        dataset = dataset.rename_column(id_column, "id")
     
    # Remove NA rows
    dataset = dataset.filter(lambda row: pd.notnull(row["text"]))
    print("Done. (1/5)")
    
    # Remove specal characters
    print("Removing special characters...")
    def remove_sp_fn(dataset):
        dataset["text"]=re.sub(r'[^a-z|A-Z|0-9|ㄱ-ㅎ|ㅏ-ㅣ|가-힣| ]+', '', str(dataset["text"]))
        return dataset
    
    dataset = dataset.map(remove_sp_fn)
    print("Done. (2/5)")
    
    # Tokenize
    print("Tokenining the text column...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side = 'left')
    def tokenize_fn(dataset):
        tokenized_batch = tokenizer(dataset["text"], padding="max_length", truncation=True)
        return tokenized_batch
    
    dataset = dataset.map(tokenize_fn, batched=True)
    tokenizer.save_pretrained(custom_tokenizer_dir)
    print("Done. (3/5)")
    
    # train-evaluation-test split
    print("Spliting train-evaluation-test set...")
    train_dataset = dataset["train"].shuffle(seed=seed).select(range(0,math.floor(len(dataset["train"])*train_proportion)))
    eval_dataset = dataset["train"].shuffle(seed=seed).select(range(math.floor(len(dataset["train"])*train_proportion), len(dataset["train"])))
    test_dataset = dataset["test"]
    print("Done. (4/5)")
    
    # Add oversampling for imbalanced dataset
    print("Applying oversampling to balance the dataset...")
    # Extract the labels from the train_dataset
    train_labels = np.array(train_dataset['label'])

    # Create a RandomOverSampler instance
    ros = RandomOverSampler(random_state=seed)

    # Fit the sampler on the training data
    train_indices, train_labels_resampled = ros.fit_resample(np.arange(len(train_dataset)).reshape(-1, 1), train_labels)

    # Update the train_dataset with the resampled indices
    train_dataset = train_dataset.select(train_indices.flatten().tolist())
    print("Done. (5/5)")
    
    return train_dataset, eval_dataset, test_dataset

def modeling(train_dataset: Dataset,
             eval_dataset: Dataset,
             model_name: str,
             num_gpus: int,
             num_cpus: int,
             seed: int,
             output_dir: str = './output',
             logging_dir: str = "./logs",
             do_hpo: bool = False,
             std: float = 0.1,
             n_trials: int = 5,
             patience: int = 3,
             hpo_result_dir: str = "./hpo-results",
             hpo_result_dir_subfolder_name: str = 'tune_transformer_pbt',
             custom_model_dir: str = "my_result"
             ) -> Type[Trainer]:
    if not isinstance(train_dataset, Dataset):
        raise TypeError(f"Values in `train_dataset` should be of type `Dataset` but got type '{type(train_dataset)}'")
    elif not isinstance(eval_dataset, Dataset):
        raise TypeError(f"Values in `eval_dataset` should be of type `Dataset` but got type '{type(eval_dataset)}'")
        
    train_dataset = train_dataset.remove_columns('id')
    eval_dataset = eval_dataset.remove_columns('id')
        
    # Load the model 
    def _model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            output_attentions = False,
            output_hidden_states = False
            )

    # Define metrics to use for evaluation
    def _compute_metrics(eval_pred):
        metric1 = load_metric("accuracy")
        metric2 = load_metric("f1")
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = metric1.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = metric2.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": accuracy, "f1": f1, "objective": accuracy+f1}

    # Default: batch size = 64, evaluate every 50 steps
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=2e-5, # config
        weight_decay=0.1, # config
        adam_beta1=0.1, # config
        adam_beta2=0.1, # config
        adam_epsilon=1.5e-06, # config
        num_train_epochs=15, # config
        max_steps=-1,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,  # config
        warmup_steps=0,
        logging_dir=logging_dir,
        save_strategy="steps",
        no_cuda=num_gpus <= 0, 
        seed=seed,  # config
        bf16=False, # Need torch>=1.10, Ampere GPU with cuda>=11.0
        fp16=True,
        tf32=True, 
        eval_steps = 50,
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model="objective", # f1 + acc
        report_to="none",
        skip_memory_metrics=True,
        gradient_checkpointing=True
        )
    
    # Calculate class weights
    train_labels = np.array(train_dataset["label"])
    class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train_labels), y = train_labels)
    weights = torch.tensor(class_weights, dtype = torch.float)
    
    # Define device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else: 
        device = torch.device("cpu")

    # Customize trainer class to apply class weights
    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.get("labels")
            # forward pass
            outputs = model(**inputs)
            logits = outputs.get("logits")
            # compute custom loss
            weight = weights.to(device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model_init=_model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=_compute_metrics,
        )
    
    if do_hpo == True:
    
        # Initialize Ray
        ray.shutdown()
        ray.init(log_to_driver=False, ignore_reinit_error=True, num_cpus=num_cpus, num_gpus=num_gpus, include_dashboard=False)

        # Fix batch_size in each trial
        tune_config = {
            "per_device_eval_batch_size": 16,
            "per_device_train_batch_size": 16,
            "max_steps": -1
        }

        # PBT schduler
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="objective",
            mode="max",
            perturbation_interval=1,
            hyperparam_mutations={
                "num_train_epochs": tune.randint(2, 20),
                "seed": tune.randint(1, 9999),
                "weight_decay": tune.uniform(0.0, 0.3),
                "learning_rate": tune.uniform(1e-5, 5e-5),
                "warmup_ratio": tune.uniform(0.0, 0.3),
                "adam_beta1": tune.loguniform(1e-2, 1),
                "adam_beta2": tune.loguniform(1e-3, 1),
                "adam_epsilon": tune.loguniform(1e-8, 1e-5),
            }, 
        )

        # Define columns to report
        reporter = CLIReporter(
            parameter_columns={
                "weight_decay": "w_decay",
                "learning_rate": "lr",
                "per_device_train_batch_size": "train_bs/gpu",
                "num_train_epochs": "num_epochs",
            },
            metric_columns=["eval_f1", "eval_accuracy", "eval_objective", "eval_loss", "epoch", "training_iteration"]
        )

        # Early stopping
        stopper = tune.stopper.ExperimentPlateauStopper(metric="objective", 
                                                        std=std,
                                                        top=n_trials,
                                                        mode="max",
                                                        patience=patience
                                                        )

        # HPO
        hpo_result = trainer.hyperparameter_search(
            hp_space = lambda _: tune_config,
            direction = "maximize",
            backend="ray",
            reuse_actors = True,
            n_trials=n_trials,
            resources_per_trial={"cpu": num_cpus, "gpu": num_gpus},
            scheduler=scheduler,
            keep_checkpoints_num=1,
            checkpoint_score_attr="training_iteration",
            stop=stopper,
            progress_reporter=reporter,
            local_dir=hpo_result_dir,
            name=hpo_result_dir_subfolder_name,
            log_to_file=True,
        )
        for n, v in hpo_result.hyperparameters.items():
            setattr(trainer.args, n, v)
    else:
        pass
    
    return trainer

def evaluation(trainer: Trainer, 
               eval_dataset: Dataset,
               text_column_name: str
               ) -> pd.DataFrame:
    if not isinstance(eval_dataset, Dataset):
        raise TypeError(f"Values in `eval_dataset` should be of type `Dataset` but got type '{type(eval_dataset)}'")
        
    # id column
    eval_dataset_id = eval_dataset
    eval_dataset_id = eval_dataset_id.remove_columns(['text', 'label', 'input_ids', 'attention_mask'])
    eval_dataset_id = eval_dataset_id.to_pandas()
    
    # Add ID to the result after performing prediction with eval data
    eval_dataset = eval_dataset.remove_columns('id')
    eval_pred_result = trainer.predict(test_dataset=eval_dataset)
    
    # prediction result
    pred_df = pd.DataFrame(eval_pred_result.predictions)
    pred_df.columns = [f'{text_column_name}_pred_0', f'{text_column_name}_pred_1']
    
    # classification result
    cls_label = list(eval_pred_result.label_ids)
    cls_pred = list(map(lambda x: x.index(max(x)), eval_pred_result.predictions.tolist()))
    
    eval_result_df = pd.concat([pd.DataFrame(eval_dataset_id),
                                pd.DataFrame(eval_dataset['text']), 
                                pd.DataFrame(pred_df), 
                                pd.DataFrame(cls_label),
                                pd.DataFrame(cls_pred)],
                               axis=1)
                               
    eval_result_df.columns = ['id', 'text', f'{text_column_name}_pred_0', f'{text_column_name}_pred_1', 'label', 'pred']
    
    return eval_result_df
