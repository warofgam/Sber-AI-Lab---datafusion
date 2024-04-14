mkdir -p data
mkdir -p predicts
mkdir -p embeddings
mkdir -p models
cd data/

curl -OL https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/DataFusion2024/Data/clients.csv
curl -OL https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/DataFusion2024/Data/train.csv
curl -OL https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/DataFusion2024/Data/report_dates.csv
curl -OL https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/DataFusion2024/Data/transactions.csv.zip

unzip transactions.csv.zip

rm transactions.csv.zip

cd ../