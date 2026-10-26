"""Download the public datasets: UCI Online News Popularity and TweetEval sentiment."""
import io
import urllib.request
import zipfile

from . import config as C

UCI = "https://archive.ics.uci.edu/static/public/332/online+news+popularity.zip"
TWEETEVAL = "https://huggingface.co/datasets/cardiffnlp/tweet_eval/resolve/main/sentiment/{split}-00000-of-00001.parquet"


def main():
    news = C.EXTERNAL / "uci_news"
    if not (news / "OnlineNewsPopularity" / "OnlineNewsPopularity.csv").exists():
        news.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(UCI, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(news)
        print("downloaded UCI Online News Popularity")
    tw = C.EXTERNAL / "tweet_eval"
    tw.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        out = tw / f"{split}.parquet"
        if not out.exists():
            urllib.request.urlretrieve(TWEETEVAL.format(split=split), out)
            print(f"downloaded TweetEval {split}")
    print("external datasets ready")


if __name__ == "__main__":
    main()
