# Dataset

Dataset source: UCI Machine Learning Repository, Phishing Websites (CC BY 4.0).
https://archive.ics.uci.edu/dataset/327/phishing+websites

- `python train.py` downloads the data through `ucimlrepo` (id=327).
- If the UCI server is unreachable, it falls back to `data/raw/phishing_websites_raw.csv`
  (11,055 rows x 30 features + `Result`, identical in shape and class counts to the UCI dataset).
- Cleaned and test-sample CSVs are generated here and ignored by Git by default.

Citation: Mohammad, R., Thabtah, F., McCluskey, L. (2012/2014). Phishing Websites. UCI Machine Learning Repository.
