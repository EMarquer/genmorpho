try:
    from . import train_autoencoder
except ImportError:
    import train_autoencoder
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH


if __name__ == "__main__":
    from argparse import ArgumentParser

    language_dataset = [(lang, "2016") for lang in SIG2016_LANGUAGES] + [(lang, "2019") for lang in SIG2019_HIGH]
    excluded_language_dataset = [("uzbek", "2019")] # less than 1000 words
    for language, dataset in excluded_language_dataset:
        language_dataset.remove((language, dataset))

    for data_seed_id in range(5):
        for model_seed_id in range(10):
            for language, dataset in language_dataset:
                str_args=f"-Vm {model_seed_id} -Vs {data_seed_id} -l {language} -d {dataset} --skip --max_epochs 100 --gpus=-1"
                print(f"Running for `{str_args}`")

                # argument parsing
                parser = ArgumentParser()
                parser, model_parser, dataset_parser, seed_parser = train_autoencoder.add_argparse_args(parser)

                args = parser.parse_args(str_args.split())

                # handle version details
                train_autoencoder.fix_args(args)

                train_autoencoder.main(args)