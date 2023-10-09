try:
    from .utils.autoload import load_genmorpho
    from .utils.data import collate_words
except ImportError:
    from utils.autoload import load_genmorpho
    from utils.data import collate_words
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from siganalogies import SIG2016_LANGUAGES, SIG2019_HIGH

# plot autoencoder
if __name__ == "__main__":
    dfs = []
    for dataset, language in [("2016", lang) for lang in SIG2016_LANGUAGES] + [("2019", lang) for lang in SIG2019_HIGH if lang != "uzbek"]:
        
        file_name=f"logs/ae/{dataset}/{language}/summary.csv"
        df = pd.read_csv(file_name).sort_values(by="POSIX_timestamp", axis=0).drop_duplicates(["split_seed", "model_seed", "language"], keep="last")
        #print(file_name, len(df.index))
        df = df[:50]
        dfs.append(df)
    
    full_df = pd.concat(dfs, axis=0)
    full_df["Language"] = full_df["language"].str.capitalize()
    full_df=full_df.sort_values(by="Language", axis=0)
    full_df["Dataset"] = "Sigmorphon " + full_df["dataset"].astype(str)
    full_df["Dataset"][full_df["Dataset"] == "Sigmorphon 2016"] = "Sigmorphon 2016 + Japanese Bigger Analogy Test Set"
    print(len(full_df.index), "model plotted")

    # cer
    fig = plt.figure(figsize=(12,4))
    ax: axes.Axes = sns.barplot(data=full_df, x="Language", hue="Dataset", y="gen_max_cer", errorbar="se")
    ax.set_ylabel("Character Edit Rate (CER)")
    ax.set_title("Autoencoder CER at test time using inference mode, lower is better (error bars are standard error of the mean)")
    ax.figure.autofmt_xdate(rotation=60)
    plt.tight_layout()
    plt.savefig("figs/cer.png")

    # char acc
    fig = plt.figure(figsize=(12,4))
    ax: axes.Axes = sns.barplot(data=full_df, x="Language", hue="Dataset", y="gen_max_char_accuracy", errorbar="se")
    ax.set_ylabel("Character accuracy")
    ax.set_title("Autoencoder character accuracy at test time using inference mode, higher is better (error bars are standard error of the mean)")
    ax.figure.autofmt_xdate(rotation=60)
    plt.tight_layout()
    plt.savefig("figs/char_acc.png")

    # Word acc
    fig = plt.figure(figsize=(12,4))
    ax: axes.Axes = sns.barplot(data=full_df, x="Language", hue="Dataset", y="gen_max_word_accuracy", errorbar="se")
    ax.set_ylabel("Word accuracy")
    ax.set_title("Autoencoder word accuracy at test time using inference mode, higher is better (error bars are standard error of the mean)")
    ax.figure.autofmt_xdate(rotation=60)
    plt.tight_layout()
    plt.savefig("figs/word_acc.png")

    # Word acc as table
    #print(full_df[["Language", "Dataset", "gen_max_word_accuracy"]].groupby(["Language", "Dataset"]).mean().sort_values("gen_max_word_accuracy"))


# plot ensemble autoencoder
if __name__ == "__main__":
    dfs = []
    for dataset, language in [("2016", lang) for lang in SIG2016_LANGUAGES] + [("2019", lang) for lang in SIG2019_HIGH if lang != "uzbek"]:
        try:
            file_name=f"logs/ensemble_ae/{dataset}/{language}/summary.csv"
            df = pd.read_csv(file_name).sort_values(by="POSIX_timestamp", axis=0).drop_duplicates(["split_seed", "language"], keep="last")
            #print(file_name, len(df.index))
            df = df[:5]
            dfs.append(df)
        except FileNotFoundError:
            pass
    
    full_df = pd.concat(dfs, axis=0)
    full_df["Language"] = full_df["language"].str.capitalize()
    full_df=full_df.sort_values(by="Language", axis=0)
    full_df["Dataset"] = "Sigmorphon " + full_df["dataset"].astype(str)
    full_df["Dataset"][full_df["Dataset"] == "Sigmorphon 2016"] = "Sigmorphon 2016 + Japanese Bigger Analogy Test Set"
    print(len(full_df.index), "model plotted")

    # cer
    fig = plt.figure(figsize=(12,4))
    ax: axes.Axes = sns.barplot(data=full_df, x="Language", hue="Dataset", y="gen_max_cer", errorbar="se")
    ax.set_ylabel("Character Edit Rate (CER)")
    ax.set_title("Ensemble autoencoder CER at test time using inference mode, lower is better (error bars are standard error of the mean)")
    ax.figure.autofmt_xdate(rotation=60)
    plt.tight_layout()
    plt.savefig("figs/ensemble_cer.png")

    # Word acc
    fig = plt.figure(figsize=(12,4))
    ax: axes.Axes = sns.barplot(data=full_df, x="Language", hue="Dataset", y="gen_max_word_accuracy", errorbar="se")
    ax.set_ylabel("Word accuracy")
    ax.set_title("Ensemble autoencoder word accuracy at test time using inference mode, higher is better (error bars are standard error of the mean)")
    ax.figure.autofmt_xdate(rotation=60)
    plt.tight_layout()
    plt.savefig("figs/ensemble_word_acc.png")