try:
    from . import test_autoencoder_ensemble
except ImportError:
    import test_autoencoder_ensemble
import pickle
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH


if __name__ == "__main__":
    from argparse import ArgumentParser

    language_dataset = [(lang, "2016") for lang in SIG2016_LANGUAGES] + [(lang, "2019") for lang in SIG2019_HIGH]
    excluded_language_dataset = [("uzbek", "2019")] # less than 1000 words
    for language, dataset in excluded_language_dataset:
        language_dataset.remove((language, dataset))

    for data_seed_id in range(5):
        for language, dataset in language_dataset:
            try:
                str_args=f"-Vs {data_seed_id} -l {language} -d {dataset} --skip --max_epochs 100"
                print(f"Running for `{str_args}`")

                # argument parsing
                parser = ArgumentParser()
                parser, dataset_parser, seed_parser = test_autoencoder_ensemble.add_argparse_args(parser)

                args = parser.parse_args(str_args.split())

                # handle version details
                test_autoencoder_ensemble.fix_args(args)

                test_autoencoder_ensemble.main(args)
            except pickle.UnpicklingError:
                pass