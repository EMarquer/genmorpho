import csv
import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

def load_model_data(dataset, language, model_seed_id=0, data_seed_id=0, root=f"{PROJECT_ROOT}/logs/ae", map_location="cpu"):
    folder_name = f"{root}/{dataset}/{language}/model{model_seed_id}-data{data_seed_id}"

    with open(f"{folder_name}/summary.csv", 'r') as f:
        reader = csv.reader(f)
        iterable = iter(reader)
        labels = next(iterable)
        values = next(iterable)

        dict_data = dict(zip(labels, values))

    best_model = dict_data["best_model"]

    import torch
    try:
        model_data = torch.load(best_model, map_location=map_location)
    except FileNotFoundError:
        model_data = torch.load(os.path.join(PROJECT_ROOT, best_model), map_location=map_location)
    model_data["training_data"] = dict_data
    return model_data

def load_pytorch(dataset, language, model_seed_id=0, data_seed_id=0, root=f"{PROJECT_ROOT}/logs/ae", map_location="cpu"):
    model_data = load_model_data(dataset, language, model_seed_id=model_seed_id, data_seed_id=data_seed_id, root=root, map_location=map_location)
    encoder = model_data["hyper_parameters"].pop("encoder")

    import morpho_gen
    model = morpho_gen.AutoEncoder(len(encoder.id_to_char), padding_index=encoder.PAD_ID, **model_data["hyper_parameters"])
    state_dict = {
        k[3:]: v
        for k,v in model_data["state_dict"].items()
        if k.startswith("ae.")
    }
    model.load_state_dict(state_dict)
    return model, encoder


def load_lightning(dataset, language, model_seed_id=0, data_seed_id=0, root=f"{PROJECT_ROOT}/logs/ae", map_location="cpu"):
    model_data = load_model_data(dataset, language, model_seed_id=model_seed_id, data_seed_id=data_seed_id, root=root, map_location=map_location)
    
    import train_autoencoder
    lightning_model = train_autoencoder.LightningAutoEncoder(**model_data["hyper_parameters"])
    lightning_model.load_state_dict(model_data["state_dict"])
    return lightning_model

if __name__ == "__main__":
    #load_pytorch("2016", "finnish")
    from utils.data import collate_words
    ae, char_encoder = load_pytorch(dataset="2019", language="english", model_seed_id=0, data_seed_id=0)
    word = "word"
    encoded_word = char_encoder.encode(word)
    encoded_word = collate_words([encoded_word],
        bos_id = char_encoder.BOS_ID,
        eos_id = char_encoder.EOS_ID,
        pad_id= char_encoder.PAD_ID)
    embedding = ae.encoder(encoded_word)
    generated = ae.decoder.generate(embedding,
        initial_character=char_encoder.BOS_ID,
        stop_character=char_encoder.EOS_ID,
        pad_character=char_encoder.PAD_ID,
        max_size=64,
        sample=False)
    print(char_encoder.decode(generated))


    from utils.data import collate_words
    lightning_model = load_lightning(dataset="2019", language="english", model_seed_id=0, data_seed_id=0)
    word = "word"
    encoded_word = lightning_model.encoder.encode(word)
    encoded_word = collate_words([encoded_word],
        bos_id = lightning_model.encoder.BOS_ID,
        eos_id = lightning_model.encoder.EOS_ID,
        pad_id= lightning_model.encoder.PAD_ID)
    embedding = lightning_model.ae.encoder(encoded_word)
    generated = lightning_model.ae.decoder.generate(embedding,
        initial_character=lightning_model.encoder.BOS_ID,
        stop_character=lightning_model.encoder.EOS_ID,
        pad_character=lightning_model.encoder.PAD_ID,
        max_size=64,
        sample=False)
    print(lightning_model.encoder.decode(generated))