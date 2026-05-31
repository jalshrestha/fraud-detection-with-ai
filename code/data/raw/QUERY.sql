-- BigQuery: pull the transaction neighbourhood for the seed address lists.
-- Replace the two arrays below with the lowercased contents of
-- data/addresses/benign_addresses.py and data/addresses/fraud_addresses.py
-- before running. Adjust the date range to control corpus size.

WITH seeds AS (
  SELECT addr FROM UNNEST([
    '0xdac17f958d2ee523a2206206994597c13d831ec7',
    -- ... paste the rest of BENIGN_ADDRESSES + FRAUD_ADDRESSES here
    '0xbb9bc244d798123fde783fcc1c72d3bb8c189413'
  ]) AS addr
)
SELECT
  `hash`,
  block_timestamp,
  from_address,
  to_address,
  value,
  gas,
  gas_price,
  receipt_status
FROM `bigquery-public-data.crypto_ethereum.transactions`
WHERE block_timestamp BETWEEN TIMESTAMP('2022-01-01') AND TIMESTAMP('2023-01-01')
  AND (from_address IN (SELECT addr FROM seeds)
       OR to_address IN (SELECT addr FROM seeds))
ORDER BY block_timestamp;
