import logging
from math import ceil
from packaging import version

from utils.data import collate_words

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

try:
    from .morpho_gen import AutoEncoder
    from .utils.logger import append_csv, to_csv
except ImportError:
    from morpho_gen import AutoEncoder
    from utils.logger import append_csv, to_csv
import torch
import torch.nn as nn
from torchmetrics import CharErrorRate, Accuracy, MeanMetric
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging


from siganalogies import dataset_factory, SIG2016_LANGUAGES, SIG2019_HIGH, CharEncoder
from torch.utils.data import DataLoader, random_split

import os
os.environ['PYTHONHASHSEED'] = str(42)
os.environ['PYTORCH_CUDA_ALLOC_CONF']="max_split_size_mb:128"
seed_everything(42, workers=True)

torch.cuda.memory.set_per_process_memory_fraction(0.9)

RANDOM_SEEDS = [42, 8564851, 706303, 248, 8994204, 7332146, 800, 3347863, 1402754, 7938707]

VERSION="debug"


class LightningAutoEncoder(pl.LightningModule):
    def __init__(self, encoder: CharEncoder, hidden_size: int, char_emb_size: int=32):
        """
        
        Param:
            hidden_size: size of the hidden layer in the encoder; word embedding is 4 times this number
        """
        super().__init__()
        self.save_hyperparameters()
        self.encoder = encoder
        self.ae = AutoEncoder(len(self.encoder.id_to_char), hidden_size, char_emb_size=char_emb_size, padding_index=self.encoder.PAD_ID)

        # all the metrics, see the README for details
        self.tf_cer = CharErrorRate()
        self.gen_max_cer = CharErrorRate()
        self.gen_rand_cer = CharErrorRate()
        self.tf_char_accuracy = Accuracy(num_classes=len(self.encoder.id_to_char), ignore_index=self.encoder.PAD_ID, multiclass=True, mdmc_average='samplewise')
        self.tf_word_accuracy = MeanMetric()
        self.gen_max_char_accuracy = Accuracy(num_classes=len(self.encoder.id_to_char), ignore_index=self.encoder.PAD_ID, multiclass=True, mdmc_average='samplewise')
        self.gen_max_word_accuracy = MeanMetric()
        self.gen_rand_char_accuracy = Accuracy(num_classes=len(self.encoder.id_to_char), ignore_index=self.encoder.PAD_ID, multiclass=True, mdmc_average='samplewise')
        self.gen_rand_word_accuracy = MeanMetric()

    def configure_optimizers(self):
        # @lightning method
        # A bunch of optimizers we tested.
        #optimizer = torch.optim.Adam(self.ae.parameters(), lr=1e-2)
        #optimizer = torch.optim.Adamax(self.ae.parameters(), lr=1e-3)  #Nope
        #optimizer = torch.optim.Adadelta(self.ae.parameters(), lr=1) #Nope
        optimizer = torch.optim.NAdam(self.ae.parameters(), lr=1e-2) #Better
        #optimizer = torch.optim.RAdam(self.ae.parameters(), lr=1e-3) #Meh
        #optimizer = torch.optim.AdamW(self.ae.parameters(), lr=1e-3) #Meh
        return optimizer
        
        #optimizer = torch.optim.SGD(self.ae.parameters(), lr=1) #Nope
        #lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        #optimizer = torch.optim.ASGD(self.ae.parameters(), lr=1e-2) #Big nope
        #return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        # @lightning method

        loss = self.ae.one_step(batch)

        # actual interesting metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss


    def validation_step(self, batch, batch_idx):
        # @lightning method

        loss = self.ae.one_step(batch)
        
        # actual interesting metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        # @lightning method
        embs = self.ae.encoder(batch)
        target = batch[:,1:]
        teacher_forcing = self.ae.decoder(embs, batch[:,:-1])
        generated_max = self.ae.decoder.generate(embs,
            initial_character=self.encoder.BOS_ID,
            stop_character=self.encoder.EOS_ID,
            pad_character=self.encoder.PAD_ID,
            max_size=max(64, target.size(-1)),
            sample=False)
        generated_rand = self.ae.decoder.generate(embs,
            initial_character=self.encoder.BOS_ID,
            stop_character=self.encoder.EOS_ID,
            pad_character=self.encoder.PAD_ID,
            max_size=max(64, target.size(-1)),
            sample=True)

        # make a copy of each output at the size of the target, except for teacher_forcing which is already at the right size
        if generated_max.size(-1) >= target.size(-1):
            generated_max_size_corrected = generated_max[:,:target.size(-1)]
        else:
            generated_max_size_corrected = nn.functional.pad(generated_max, (0, target.size(-1)-generated_max.size(-1)), value=self.encoder.PAD_ID)
        if generated_rand.size(-1) >= target.size(-1):
            generated_rand_size_corrected = generated_rand[:,:target.size(-1)]
        else:
            generated_rand_size_corrected = nn.functional.pad(generated_rand, (0, target.size(-1)-generated_rand.size(-1)), value=self.encoder.PAD_ID)

        # compute loss
        criterion = nn.CrossEntropyLoss(ignore_index=self.encoder.PAD_ID)
        self.log(f'test_loss', criterion(teacher_forcing.transpose(-1,-2), target), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # get the maximal probabilities for the teacher forcing output, and replace any data past the end of sequence character
        teacher_forcing = torch.argmax(teacher_forcing, dim=-1)
        teacher_forcing[target==self.encoder.PAD_ID] = self.encoder.PAD_ID # equivalent to the ignore index
        
        # update the accuracy metrics
        self.tf_char_accuracy.update(teacher_forcing, target)
        self.gen_max_char_accuracy.update(generated_max_size_corrected, target)
        self.gen_rand_char_accuracy.update(generated_rand_size_corrected, target)
        self.tf_word_accuracy.update((teacher_forcing==target).all(dim=-1))
        self.gen_max_word_accuracy.update((generated_max_size_corrected==target).all(dim=-1))
        self.gen_rand_word_accuracy.update((generated_rand_size_corrected==target).all(dim=-1))

        # update the character error rate metrics
        decoded_tf = self.encoder.decode(teacher_forcing, pad_char="")
        decoded_gen_max = self.encoder.decode(generated_max, pad_char="")
        decoded_gen_rand = self.encoder.decode(generated_rand, pad_char="")
        decoded_target = self.encoder.decode(target, pad_char="")
        self.tf_cer.update(decoded_tf, decoded_target)
        self.gen_max_cer.update(decoded_gen_max, decoded_target)
        self.gen_rand_cer.update(decoded_gen_rand, decoded_target)

    def test_epoch_end(self, outputs):
        # log the test metrics
        results = [
            # Teacher forcing metric
            ('tf_word_accuracy', self.tf_word_accuracy.compute()),
            ('tf_char_accuracy', self.tf_char_accuracy.compute()),
            ('tf_cer', self.tf_cer.compute()),
            # Max-value generation metric
            ('gen_max_word_accuracy', self.gen_max_word_accuracy.compute()),
            ('gen_max_char_accuracy', self.gen_max_char_accuracy.compute()),
            ('gen_max_cer', self.gen_max_cer.compute()),
            # Sampling generation metric
            ('gen_rand_word_accuracy', self.gen_rand_word_accuracy.compute()),
            ('gen_rand_char_accuracy', self.gen_rand_char_accuracy.compute()),
            ('gen_rand_cer', self.gen_rand_cer.compute()),
        ]
        for key, result in results:
            self.log(f'test_{key}', result, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def example(self, batch):
        # generate the autoencoding output for each example in the batch
        embs = self.ae.encoder(batch)
        target = batch[:,1:]
        generated_max = self.ae.decoder.generate(embs,
            initial_character=self.encoder.BOS_ID,
            stop_character=self.encoder.EOS_ID,
            pad_character=self.encoder.PAD_ID,
            max_size=max(64, target.size(-1)),
            sample=False)
        generated_rand = self.ae.decoder.generate(embs,
            initial_character=self.encoder.BOS_ID,
            stop_character=self.encoder.EOS_ID,
            pad_character=self.encoder.PAD_ID,
            max_size=max(64, target.size(-1)),
            sample=True)

        # decode the input, expected output and actual outputs
        decoded_gen_max_acc = self.encoder.decode(generated_max)
        decoded_generated_rand = self.encoder.decode(generated_rand)
        decoded_target = self.encoder.decode(target)
        decoded_input = self.encoder.decode(batch)

        # print the examples
        print("#### Examples ####")
        for i in range(len(batch)):
            print()
            print("Input:            ", decoded_input[i])
            print("Target:           ", decoded_target[i])
            print("Generated (max):  ", decoded_gen_max_acc[i])
            print("Generated (rand): ", decoded_generated_rand[i])

    def log_to_file(self, path, common_path, extra_data):
        # write the metrics and the extra_data to the CSV summary files
        row = [
            # Teacher forcing metric
            ('tf_word_accuracy', self.tf_word_accuracy.compute().item()),
            ('tf_char_accuracy', self.tf_char_accuracy.compute().item()),
            ('tf_cer', self.tf_cer.compute().item()),
            # Max-value generation metric
            ('gen_max_word_accuracy', self.gen_max_word_accuracy.compute().item()),
            ('gen_max_char_accuracy', self.gen_max_char_accuracy.compute().item()),
            ('gen_max_cer', self.gen_max_cer.compute().item()),
            # Sampling generation metric
            ('gen_rand_word_accuracy', self.gen_rand_word_accuracy.compute().item()),
            ('gen_rand_char_accuracy', self.gen_rand_char_accuracy.compute().item()),
            ('gen_rand_cer', self.gen_rand_cer.compute().item())
        ]
        row.extend(extra_data.items())
        if self.trainer.is_global_zero:
            to_csv(path, dict(row))
            append_csv(common_path, dict(row))

SPLIT_RANDOM_SEED = 42
DATALOADER_SEED = 42
def prepare_word_data(dataset_year, language, nb_word_train, nb_word_val, nb_word_test,
        batch_size = 32, force_rebuild=False, split_seed=SPLIT_RANDOM_SEED, dataloader_seed=DATALOADER_SEED, force_low=False):
    # prepare the data
    # the code might appear obscure, but there is a slightly different way of obtaining the data on Siganalogies SIG2019
    
    if not force_low and ((dataset_year == "2019" and language in SIG2019_HIGH) or (dataset_year == "2016" and language=="japanese")):
        mode = "train" if dataset_year == "2016" else "train-high"
        dataset = dataset_factory(dataset=dataset_year, language=language, mode=mode, word_encoder="char", force_rebuild=force_rebuild)
    else:
        dataset = dataset_factory(dataset=dataset_year, language=language, mode="train" if dataset_year == "2016" else "train-low", word_encoder="char", force_rebuild=force_rebuild)
        
    # grab the data wocabulary
    data = list(dataset.word_voc)

    # split the data into train validation and test
    lengths = [nb_word_train, nb_word_val, nb_word_test]
    if sum(lengths) > len(data): # we don't have enough data, so we reduce the training set
        lengths[0] = len(data) - nb_word_val - nb_word_test
    if sum(lengths) < len(data): # we have more than enough data, so there will be excess
        train_data, val_data, test_data, unused = random_split(data, [*lengths, len(dataset) - sum(lengths)],
                    generator=torch.Generator().manual_seed(split_seed))
    else: # we have just enough data, so there is no excess
        train_data, val_data, test_data = random_split(data, lengths,
                generator=torch.Generator().manual_seed(split_seed))

    print("Train, dev, test:", lengths)

    # this function gather words into a batch
    def collate(batch):
        batch = [dataset.word_encoder.encode(word) for word in batch]
        return collate_words(batch, bos_id = dataset.word_encoder.BOS_ID, eos_id = dataset.word_encoder.EOS_ID, pad_id= dataset.word_encoder.PAD_ID)


    # we create the dataloaders
    args = {
        "collate_fn": collate,
        "num_workers": 4,
        #"batch_size": batch_size,
        "persistent_workers": True
    }

    # this here is a reminder of an experiment where we trained the model with different batch size
    if isinstance(batch_size, (list, tuple)) and len(batch_size) > 0:
        train_loader = [DataLoader(train_data, batch_size=_batch_size, generator=torch.Generator().manual_seed(dataloader_seed), shuffle=True, **args)
            for _batch_size in batch_size]
        val_loader = DataLoader(val_data, batch_size=batch_size[0], **args)#, generator=g_val)
        test_loader = DataLoader(test_data, batch_size=batch_size[0], **args)#, generator=g_test)

    # we actually create the dataloaders here
    else:
        train_loader = [DataLoader(train_data, batch_size=batch_size, generator=torch.Generator().manual_seed(dataloader_seed), shuffle=True, **args)]
        val_loader = DataLoader(val_data, batch_size=batch_size, **args)#, generator=g_val)
        test_loader = DataLoader(test_data, batch_size=batch_size, **args)#, generator=g_test)

    return train_loader, val_loader, test_loader, dataset.word_encoder

def main(args):
    
    # the names defined here correspond to the files and folder of the results
    expe_group = f"ae/{args.dataset}/{args.language}"
    common_summary_file = f"logs/{expe_group}/summary.csv"
    expe_name = f"{expe_group}/model{args.model_seed_id}-data{args.split_seed_id}"
    summary_file = f"logs/{expe_name}/summary.csv"
    if args.skip and os.path.exists(summary_file):
        print(f"{summary_file} exists, skip")
        return


    logger.warning("Determinism (--deterministic True) does not guarantee reproducible results when changing the number of processes.")

    # --- Prepare the data ---
    seed_everything(RANDOM_SEEDS[args.split_seed_id], workers=True)
    train_loader, val_loader, test_loader, encoder = prepare_word_data(
        args.dataset, args.language, args.nb_word_train, args.nb_word_val, args.nb_word_test,
        [args.batch_size], args.force_rebuild, split_seed=RANDOM_SEEDS[args.split_seed_id], dataloader_seed=RANDOM_SEEDS[args.model_seed_id])

    if args.max_epochs is not None:
        args.max_epochs = ceil(args.max_epochs * args.nb_word_train / len(train_loader[0].dataset))
        logger.info(f"Scaling the number of max_epochs to {(args.nb_word_train / len(train_loader[0].dataset)):.0%}: {args.max_epochs}")

    # --- Define models ---
    seed_everything(RANDOM_SEEDS[args.model_seed_id], workers=True)
    nn = LightningAutoEncoder(encoder=encoder, hidden_size=args.emb_size//4, char_emb_size=args.char_emb_size)

    # --- Train model in two steps ---
    tb_logger = pl.loggers.TensorBoardLogger('logs/', expe_name, version=VERSION)
    checkpoint_callback=ModelCheckpoint(
        filename=f"ae-{args.dataset}-{args.language}-b{args.batch_size}-{{epoch:02d}}",
        monitor="val_loss", mode="min", save_top_k=1)
    early_stopping_callback=EarlyStopping(monitor="val_loss", patience=6)
    
    # handle version details
    def get_trainer_kwargs():
        use_spawn = False
        trainer_kwargs = dict()
        # DistributedDataParallel
        if version.parse(pl.__version__) >= version.parse("1.6.0"):
            from pytorch_lightning.strategies.ddp import DDPStrategy
            trainer_kwargs["strategy"] = DDPStrategy(find_unused_parameters=False)
        elif version.parse(pl.__version__) >= version.parse("1.5.0"):
            from pytorch_lightning.plugins import DDPSpawnPlugin, DDPPlugin
            trainer_kwargs["strategy"] = (DDPSpawnPlugin(find_unused_parameters=False) if use_spawn # causes dataloader issues
                else DDPPlugin(find_unused_parameters=False))
        else:
            from pytorch_lightning.plugins import DDPSpawnPlugin, DDPPlugin
            trainer_kwargs["plugins"] = (DDPSpawnPlugin(find_unused_parameters=False) if use_spawn # causes dataloader issues
                else DDPPlugin(find_unused_parameters=False))
        # GPUs
        if version.parse(pl.__version__) >= version.parse("1.7.0") and args.gpus:
            args.accelerator='gpu'
            args.devices=args.gpus
            args.gpus=None

        return trainer_kwargs
    
    # --- Train model ---
    # create trainer
    trainer = pl.Trainer.from_argparse_args(args,
        callbacks=[
            early_stopping_callback,
            checkpoint_callback,
            StochasticWeightAveraging(swa_lrs=1e-2)
        ],
        logger = tb_logger,
        **get_trainer_kwargs()
    )

    # train
    seed_everything(RANDOM_SEEDS[args.model_seed_id], workers=True)
    trainer.fit(nn, train_loader[0], val_loader)

    import datetime
    
    
    with torch.no_grad():
        # test
        trainer.test(nn, dataloaders=test_loader, ckpt_path=checkpoint_callback.best_model_path)
        # produce some examples
        nn.example(test_loader.collate_fn([test_loader.dataset[i] for i in range(10)]))
        # log the performance
        extra_info = {
            "dataset": args.dataset,
            "language": args.language,
            "model_seed" : RANDOM_SEEDS[args.model_seed_id],
            "split_seed" : RANDOM_SEEDS[args.split_seed_id],
            "POSIX_timestamp" : datetime.datetime.now().timestamp(),
            "best_model" : checkpoint_callback.best_model_path,
            "epochs" : trainer.current_epoch,
        }
        nn.log_to_file(summary_file, common_summary_file, extra_info)


def add_argparse_args(parser):
    # argument parsing
    parser = pl.Trainer.add_argparse_args(parser)

    model_parser = parser.add_argument_group("Model arguments")
    model_parser.add_argument('--emb_size', '-H', type=int, default=256, help='The size of the word embedding. Each hidden state and cell state of the encoder BiLSTM is 1/4 of that number, and each hidden state and cell state of the decoder LSTM is 1/2 of that number. Must be dividable by 4.')
    model_parser.add_argument('--char_emb_size', '-Hc', type=int, default=0, help='The size of the character embedding of the model. Default 0 (one hot encoding only), if used recommended 64.')

    dataset_parser = parser.add_argument_group("Dataset arguments")
    dataset_parser.add_argument('--dataset', '-d', type=str, default="2016", help='The language to train the model on.', choices=["2016", "2019"])
    dataset_parser.add_argument('--language', '-l', type=str, default="arabic", help='The language to train the model on.', choices=SIG2019_HIGH+SIG2016_LANGUAGES)
    dataset_parser.add_argument('--force_rebuild', help='Force the re-building of the dataset file.',  action='store_true')
    dataset_parser.add_argument('--nb_word_train', '-n', type=int, default=40000, help='The maximum number of words we train the model on.')
    dataset_parser.add_argument('--nb_word_val', '-v', type=int, default=500, help='The number of words we validate the model on.')
    dataset_parser.add_argument('--nb_word_test', '-t', type=int, default=500, help='The number of words we test the model on.')
    dataset_parser.add_argument('--batch_size', '-b', type=int, default=2048, help='Batch size.')
    dataset_parser.add_argument('--skip', help='Skip if such a model has been trained already. Uses the summary.csv file in the model folder to determine if the model has been trained already on the language.', action='store_true')

    seed_parser = parser.add_argument_group("Random seed arguments")
    seed_parser.add_argument('--model_seed_id', '-Vm', type=int, default=0, help='The model seed.')
    seed_parser.add_argument('--split_seed_id', '-Vs', type=int, default=0, help='The model seed.')
    
    return parser, model_parser, dataset_parser, seed_parser

def fix_args(args):
    # Different ways of asking for GPUs
    if version.parse(pl.__version__) >= version.parse("1.7.0") and args.gpus:
        args.accelerator='gpu'
        args.devices=args.gpus
        args.gpus=None

if __name__ == '__main__':
    from argparse import ArgumentParser

    # argument parsing
    parser = ArgumentParser()
    parser, model_parser, dataset_parser, seed_parser = add_argparse_args(parser)

    args = parser.parse_args()

    # handle version details
    fix_args(args)

    # start the training script
    main(args)
