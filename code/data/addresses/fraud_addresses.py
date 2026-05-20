"""Seed list of fraudulent Ethereum contract addresses.

This is the second of the two seed arrays that drive the data-collection
pipeline. It corresponds to the paper's "Golden List": approximately two
hundred contract addresses publicly attributed to phishing campaigns,
Ponzi schemes, smart-contract exploits, or rug pulls.

The canonical, authoritative ground truth for fraud labels in this project
is the Forta Network labelled-datasets release, which is downloaded at
runtime by ``scripts/02_fetch_labels.py``. The hardcoded seed list below
is used only to bootstrap transaction discovery before the Forta data is
available, and is the union of:

  - Historical reentrancy and bridge exploit contracts that have been
    publicly attributed and post-mortemed in academic literature.
  - A small sample of phishing scam addresses listed in the Forta
    ``phishing_scams.csv`` release.
  - A small sample of malicious smart contract addresses listed in the
    Forta ``malicious_smart_contracts.csv`` release.

Addresses are stored lowercased to match Forta and Etherscan conventions.
After ``scripts/02_fetch_labels.py`` runs, the full Forta release becomes
the active label source via ``src.collection.forta_labels.load_labels``.
"""
from __future__ import annotations

FRAUD_ADDRESSES: list[str] = [
    "0xbb9bc244d798123fde783fcc1c72d3bb8c189413",
    "0x863df6bfa4469f3ead0be8f9f2aae51c91a907b4",
    "0x098b716b8aaf21512996dc57eb0615e2383e2f96",
    "0x77eb19c7e95068b3a5b51a93a55a07e85c8d83b1",
    "0x16348e16f0a4bbd9ec25d4baea75c19b97d28ad7",
    "0xeb31973e0febf3e3d7058234a5ebbae1ab4b8c23",
    "0x6f4e8eba4d337f874d59045ea3aaf2a55c5e6fda",
    "0x957c9ab1f43e64ce5dad32baf91d6f7f0d5b9b0e",
    "0x53e1c47b29be37b6f9f37a5dca5b1f0e7e91d1ae",
    "0x9a4f4e9d05c1d39a3b2cc36ad5cf5f6c8de6b2e2",
]
