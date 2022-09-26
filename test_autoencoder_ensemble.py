from collections import Counter
import logging
from math import ceil
from packaging import version

logger = logging.getLogger("")#__name__)
logger.setLevel(logging.INFO)

# configure logging at the root level of lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# add the handlers to the loggers
logger.addHandler(ch)

import torch
import torch.nn as nn
from torchmetrics import CharErrorRate, MeanMetric
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

import datetime

import os
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['PYTORCH_CUDA_ALLOC_CONF']="max_split_size_mb:128"
seed_everything(42, workers=True)

torch.cuda.memory.set_per_process_memory_fraction(0.9)

VERSION="debug"

try:
    from .train_autoencoder import LightningAutoEncoder, prepare_word_data, fix_args, RANDOM_SEEDS
    from .utils.logger import to_csv, append_csv
    from .utils.autoload import load_lightning, load_model_data
except ImportError:
    from train_autoencoder import LightningAutoEncoder, prepare_word_data, fix_args, RANDOM_SEEDS
    from utils.logger import to_csv, append_csv
    from utils.autoload import load_lightning, load_model_data

class ModelEnsemble(pl.LightningModule):
    def __init__(self, models, weights=None) -> None:
        super().__init__()
        self.models = models
        if weights is None:
            self.weights = nn.Parameter(torch.ones((len(models),)))
        else:
            self.weights = nn.Parameter(torch.tensor(weights))
        self.getWeights = lambda: nn.functional.softmax(self.weights, -1)
        self.gen_cer = CharErrorRate()
        self.gen_word_accuracy = MeanMetric()
        self.gen_cer_ = .5
        self.gen_word_accuracy_ = .5

    def generate(self, batch):
        batch = batch.to(next(self.models[0].ae.parameters()).device )
        target = batch[:,1:]
        results = list()

        decoded_results = [Counter() for i in range(batch.size(-2))]
        for i, model in enumerate(self.models):
            embs = model.ae.encoder(batch)
            result = model.ae.decoder.generate(embs,
                    initial_character=model.encoder.BOS_ID,
                    stop_character=model.encoder.EOS_ID,
                    pad_character=model.encoder.PAD_ID,
                    max_size=max(64, target.size(-1)),
                    sample=False)
            decoded_result = model.encoder.decode(result, pad_char="")
            results.append(result)
            #decoded_results.append(decoded_result)
            for j in range(len(decoded_results)):
                decoded_results[j][decoded_result[j]] += self.getWeights()[i]

        #decoded_results = list(zip(*decoded_results))
        vote_results = [decoded.most_common(1)[0][0] for decoded in decoded_results]
        #voted_results = list(Counter(result).most_common(1)[0][0] for result in decoded_results)

        return decoded_results, vote_results

    def test_step(self, batch, batch_idx):
        # @lightning method
        target = batch[:,1:]
        decoded_results, voted_results = self.generate(batch)
        
        decoded_targets = self.models[0].encoder.decode(target, pad_char="")
        self.gen_cer(voted_results, decoded_targets)
        
        for voted_result, decoded_target in zip(voted_results, decoded_targets):
            self.gen_word_accuracy.update(torch.tensor([1. if voted_result == decoded_target else 0.], device=batch.device))
            #self.gen_word_accuracy.update((1. if voted_result == decoded_target else 0.))
        
    def test_epoch_end(self, outs):
        # log epoch metric
        self.gen_cer_ = self.gen_cer.compute().item()
        self.gen_word_accuracy_ = self.gen_word_accuracy.compute().item()
        self.log('test_gen_cer', self.gen_cer_)
        self.log('test_gen_word_acc', self.gen_word_accuracy_)

    def example(self, batch):
        target = batch[:,1:]
        decoded_results, voted_results = self.generate(batch)
        decoded_input = self.models[0].encoder.decode(batch, pad_char="")
        decoded_target = self.models[0].encoder.decode(target, pad_char="")

        print("#### Examples ####")
        for i in range(len(batch)):
            print()
            print("Input:            ", decoded_input[i])
            print("Target:           ", decoded_target[i])
            print("Generated (voted):  ", voted_results[i])
            print("Generated (all): ", decoded_results[i])

from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH
SPLIT_RANDOM_SEED = 42
DATALOADER_SEED = 42

N_MODELS = 10
def make_ensemble(dataset, language, split_seed_id=0, initialize_weights=True):
    models = []
    weights = []
    for i in range(N_MODELS):
        model_data = load_model_data(dataset, language, model_seed_id=i, data_seed_id=split_seed_id)
        weights.append(1-float(model_data["training_data"]["gen_max_cer"]))
        #weights.append(1-float(model_data["training_data"]["gen_max_word_accuracy"]))
        models.append(load_lightning(dataset, language, model_seed_id=i, data_seed_id=split_seed_id))
    if initialize_weights:
        nns = ModelEnsemble(models, weights=weights)
    else:
        nns = ModelEnsemble(models)
    return nns

def main(args):
    expe_group = f"ensemble_ae/{args.dataset}/{args.language}"
    common_summary_file = f"logs/{expe_group}/summary.csv"
    expe_name = f"{expe_group}/data{args.split_seed_id}"
    summary_file = f"logs/{expe_name}/summary.csv"
    if args.skip and os.path.exists(summary_file):
        print(f"{summary_file} exists, skip")
        return

    # --- Prepare the data ---
    seed_everything(RANDOM_SEEDS[args.split_seed_id], workers=True)
    train_loader, val_loader, test_loader, encoder = prepare_word_data(
        args.dataset, args.language, args.nb_word_train, args.nb_word_val, args.nb_word_test,
        [args.batch_size], args.force_rebuild, split_seed=RANDOM_SEEDS[args.split_seed_id])

    if args.max_epochs is not None:
        args.max_epochs = ceil(args.max_epochs * args.nb_word_train / len(train_loader[0].dataset))
        logger.info(f"Scaling the number of max_epochs to {(args.nb_word_train / len(train_loader[0].dataset)):.0%}: {args.max_epochs}")

    # --- Define models ---
    nns = make_ensemble(args.dataset, args.language, split_seed_id=args.split_seed_id)
    
    # handle version details
    def fix_args(args):
        # GPUs
        if version.parse(pl.__version__) >= version.parse("1.7.0") and args.gpus:
            args.accelerator='gpu'
            args.devices=args.gpus
            args.gpus=None
    fix_args(args)

    def get_trainer_kwargs(find_unused_parameters=False):
        use_spawn = False
        trainer_kwargs = dict()
        
        # DistributedDataParallel
        if version.parse(pl.__version__) >= version.parse("1.6.0"):
            from pytorch_lightning.strategies.ddp import DDPStrategy
            trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=find_unused_parameters)
        elif version.parse(pl.__version__) >= version.parse("1.5.0"):
            from pytorch_lightning.plugins import DDPSpawnPlugin, DDPPlugin
            trainer_kwargs["strategy"] = (DDPSpawnPlugin(find_unused_parameters=find_unused_parameters) if use_spawn # causes dataloader issues
                else DDPPlugin(find_unused_parameters=find_unused_parameters))
        else:
            from pytorch_lightning.plugins import DDPSpawnPlugin, DDPPlugin
            trainer_kwargs["plugins"] = (DDPSpawnPlugin(find_unused_parameters=find_unused_parameters) if use_spawn # causes dataloader issues
                else DDPPlugin(find_unused_parameters=find_unused_parameters))
        # GPUs
        if version.parse(pl.__version__) >= version.parse("1.7.0") and args.gpus:
            args.accelerator='gpu'
            args.devices=args.gpus
            args.gpus=None

        return trainer_kwargs

    # --- Generate ensemble model ---
    tb_logger = pl.loggers.TensorBoardLogger('logs/', expe_name, version=VERSION)
    trainer = pl.Trainer.from_argparse_args(args,
        logger = tb_logger,
        **get_trainer_kwargs(True)
    )
    seed_everything(RANDOM_SEEDS[args.split_seed_id], workers=True)

    print("Models coefficients:")
    print(nns.getWeights())
    with torch.no_grad():
        trainer.test(nns, dataloaders=test_loader)#, ckpt_path=checkpoint_callback.best_model_path)
        
        if __name__ == '__main__':
            nns.example(test_loader.collate_fn([test_loader.dataset[i] for i in range(5)]))
        
        row = {
            "dataset": args.dataset,
            "language": args.language,
            "split_seed" : RANDOM_SEEDS[args.split_seed_id],
            "POSIX_timestamp" : datetime.datetime.now().timestamp(),
            "epochs" : trainer.current_epoch,
            "gen_max_word_accuracy": nns.gen_word_accuracy_,
            "gen_max_cer": nns.gen_cer_,
        }
        to_csv(summary_file, dict(row))
        append_csv(common_summary_file, dict(row))
        


def add_argparse_args(parser):
    # argument parsing
    parser = pl.Trainer.add_argparse_args(parser)

    dataset_parser = parser.add_argument_group("Dataset arguments")
    dataset_parser.add_argument('--dataset', '-d', type=str, default="2016", help='The language to train the model on.', choices=["2016", "2019"])
    dataset_parser.add_argument('--language', '-l', type=str, default="arabic", help='The language to train the model on.', choices=SIG2019_HIGH+SIG2016_LANGUAGES)
    dataset_parser.add_argument('--force_rebuild', help='Force the re-building of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--nb_word_train', '-n', type=int, default=40000, help='The maximum number of words we train the model on.')
    dataset_parser.add_argument('--nb_word_val', '-v', type=int, default=500, help='The number of words we validate the model on.')
    dataset_parser.add_argument('--nb_word_test', '-t', type=int, default=500, help='The number of words we test the model on.')
    dataset_parser.add_argument('--batch_size', '-b', type=int, default=2048, help='Batch size.')
    dataset_parser.add_argument('--skip', help='Skip if such a model has been trained already. Uses the summary.csv file in the model folder to determine if the model has been trained already on the language.', action='store_true')

    seed_parser = parser.add_argument_group("Dataset arguments")
    seed_parser.add_argument('--split_seed_id', '-Vs', type=int, default=0, help='The model seed.')
    
    return parser, dataset_parser, seed_parser

if __name__ == '__main__':
    from argparse import ArgumentParser

    # argument parsing
    parser = ArgumentParser()
    parser, dataset_parser, seed_parser = add_argparse_args(parser)

    args = parser.parse_args()


    # handle version details
    fix_args(args)

    # start the training script
    main(args)
