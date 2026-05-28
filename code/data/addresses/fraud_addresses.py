"""Verified fraudulent Ethereum contract addresses from DeFiHackLabs.

All 115 addresses are victim protocol contracts extracted from DeFiHackLabs
PoC files (// Vulnerable Contract comments pointing to etherscan.io).
Each contract has verified Solidity source on Etherscan AND on-chain
transaction history, satisfying both the CodeBERT and GraphSAGE encoders
in the fusion model.

Source: https://github.com/SunWeb3Sec/DeFiHackLabs
Verified: Etherscan API V2, 2026-05-20
"""
from __future__ import annotations

FRAUD_ADDRESSES: list[str] = [
    # --- 2017 ---
    "0xbec591de75b8699a3ba52f073428822d0bfc0d7e",  # Wallet (Parity first hack)

    # --- 2018 ---
    "0x55f93985431fc9304077687a35a1ba103dc1e081",  # SMT (SmartMesh)
    "0xf91546835f756da0c10cfa0cda95b15577b84aa7",  # LedgerChannel (SpankChain)

    # --- 2020 ---
    "0xb983e01458529665007ff7e0cddecdb74b967eb6",  # LoanToken (bzx)
    "0xde744d544a9d768e96c21b5f087fc54b776e9b25",  # LoanTokenLogicWeth (bzx)

    # --- 2021 ---
    "0xacd43e627e64355f1861cec6d3a6688b31a6f952",  # yVault (Yearn ydai)
    "0xc4ff55a4329f84f9bf0f5619998ab570481ebb48",  # SorbettoFragola (Popsicle)

    # --- 2022 ---
    "0x39360ac1239a0b98cb8076d4135d0f72b7fd9909",  # XNFT (XCarnival)
    "0xae461ca67b15dc8dc81ce7615e0320da1a9ab8d5",  # UniswapV2Pair
    "0xe39fd820b58f83205db1d9225f28105971c3d309",  # EFLeverVault
    "0x007fe7c498a2cf30971ad8f2cbc36bd14ac51156",  # BondFixedExpiryTeller (OlympusDao)
    "0x48d118c9185e4dbafe7f3813f8f29ec8a6248359",  # LockToken (TeamFinance)
    "0xd2869042e12a3506100af1d192b5b04d65137941",  # StaxLPStaking (Templedao)
    "0x418c24191ae947a78c99fdc0e45a1f96afb254be",  # Token (Uerii)
    "0x8f9036732b9aa9b82d8f35e54b71faeb2f573e2f",  # DaoModule (XaveFinance)
    "0xf2919d1d80aff2940274014bef534f7791906ff2",  # JAY

    # --- 2023 ---
    "0x31a4f372aa891b46ba44dc64be1d8947c889e9c6",  # Shoco
    "0x765b8d7cd8ff304f796f4b6fb1bcf78698333f6d",  # ExchangeBetweenPools
    "0xaf274e912243b19b882f02d731dacd7cd13072d0",  # StrategyDAICurve (CompounderFinance)
    "0xa5564a2d1190a141cac438c9fde686ac48a18a79",  # ZeroXStargateLPSwapper (MIMSpell)
    "0x85018cf6f53c8bbd03c3137e71f4fca226cda92c",  # ApeStaking (Pawnfi)
    "0x9f72dc67cec672bb99e3d02cbea0a21536a2b657",  # InitializableImmutableAdminUpgradeabilityProxy (Sturdy)
    "0x46bea99d977f269399fb3a4637077bb35f075516",  # LendingPool (Sturdy)
    "0x3ae354d7e49039ccd582f1f3c9e65034ffd17bad",  # Vault (ArcadiaFi)
    "0x8189afbe7b0e81dae735ef027cd31371b3974feb",  # Bean (AzukiDAO)
    "0xb0f8fe96b4880adbdede0ddf446bd1e7ef122c4e",  # CErc20Delegator (Bao)
    "0xf169bd68ed72b2fdc3c9234833197171aa000580",  # TransparentUpgradeableProxy (CIVNFT)
    "0x369cbc5c6f139b1132d3b91b87241b37fc5b971f",  # ConicPoolV2 (Conic02)
    "0xbb787d6243a8d450659e09ea6fd82f1c859691e9",  # ConicEthPool (Conic)
    "0x6326debbaa15bcfe603d831e7d75f4fc10d9b43e",  # Vyper_contract (Curve01)
    "0x8301ae4fc9c624d1d396cbdaa1ed877821d7c511",  # Vyper_contract (Curve02)
    "0x9210f1204b5a24742eba12f710636d76240df3d0",  # AaveLinearPool (Balancer)
    "0x786b374b5eef874279f4b7b4de16940e57301a58",  # Vyper_contract (CurveBurner)
    "0x4ae2cd1f5b8806a973953b76f9ce6d5fab9cdcfd",  # EHIVE
    "0x863e572b215fd67c855d973f870266cf827aea5e",  # EFVault (EarningFram)
    "0x4306b12f8e824ce1fa9604bbd88f2ad4f0fe3c54",  # Uwerx
    "0xb40b6608b2743e691c9b54ddbdee7bf03cd79f1c",  # UZD (Zunami)
    "0x29d2bcf0d70f95ce16697e645e2b76d218d66109",  # OxODexPool
    "0xd3c41c85be295607e8ea5c58487ec5894300ee67",  # PointFarm (uniclyNFT)
    "0xbaa87546cf87b5de1b0b52353a86792d40b8ba70",  # ERC1967Proxy (Astrid)
    "0xae60ac8e69414c2dc362d0e6a03af643d1d85b92",  # DePayRouterV1
    "0x53fbcada1201a465740f2d64ecdf6fac425f9030",  # InitializableImmutableAdminUpgradeabilityProxy (Hopelend)
    "0x84524baa1951247b3a2617a843e6ece915bb9674",  # WiseLending
    "0x2033b54b6789a963a02bfcbd40a46816770f1161",  # UniswapV2Pair
    "0xa44e79a2c9a8965e7a6fa77bf0ca8faf50e6c73e",  # FarmingLPToken (Burntbubba)
    "0xfd7b111aa83b9b6f547e617c7601efd997f64703",  # Pool (KyberSwap)
    "0xfd11aba71c06061f446ade4eec057179f19c23c4",  # Pool (MahaLend)
    "0x9ab6b21cdf116f611110b048987e58894786c244",  # InterestRatePositionManager (Raft)
    "0xc538d17a6aacc5271be5f51b891e2e92c8187edd",  # FlooringPeriphery (FloorProtocol)
    "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b",  # Unitroller (GoodCompound)
    "0x40c31236b228935b0329eff066b1ad96e319595e",  # L1ChugSplashProxy (HYPR)
    "0xc310e760778ecbca4c65b6c559874757a4c4ece0",  # BatchSwap (NFTTrader)
    "0x2405913d54fc46eeaf3fb092bfb099f46803872f",  # ERC721LendingPool02 (PineProtocol)
    "0x4b0e9a7da8bab813efae92a6651019b8bd6c0a29",  # TokenERC20 (TIME)
    "0x7f3fe9d492a9a60aebb06d82cba23c6f32cad10b",  # LoanToken (bZx)

    # --- 2024 ---
    "0x1bf68a9d1eaee7826b3593c20a0ca93293cb489a",  # EthVault (OrbitChain)
    "0x3a23f943181408eac424116af7b7790c94cb97a5",  # SocketGateway
    "0xcc5fda5e3ca925bd0bb428c8b2669496ee43067e",  # WrappedTokenSwapperImpl (SocketGateway)
    "0x37e49bf3749513a02fa535f0cbc383796e8107e4",  # WiseLending
    "0xffadb0bba4379dfabfb20ca6823f6ec439429ec2",  # Comptroller (BlueberryProtocol)
    "0x50ce56a3239671ab62f185704caedf626352741e",  # UniswapAnchoredView (CompoundUni)
    "0x2c7112245fc4af701ebf90399264a7e89205dad4",  # TransparentUpgradeableProxy (DN404)
    "0xb57e874082417b66877429481473cf9fcd8e0b8a",  # DeezNutz
    "0x732276168b421d4792e743711e1a48172ea574a2",  # UniswapV3Pool (Miner)
    "0xe4764f9cd8ecc9659d3abf35259638b20ac536e4",  # ParticleExchange
    "0xfe380fe1db07e531e3519b9ae3ea9f7888ce20c6",  # RuggedMarket (RuggedArt)
    "0x2b9dc65253c035eb21778cb3898eab5a0ada0cce",  # XTokenWrapper (SwarmMarkets)
    "0x8584ddbd1e28bca4bc6fb96bafe39f850301940e",  # JuiceStaking
    "0xd3f64baa732061f8b3626ee44bab354f854877ac",  # TransparentUpgradeableProxy (UnizenIO)
    "0xbc452fdc8f851d7c5b72e1fe74dfb63bb793d511",  # ClaimCampaigns (HedgeyFinance)
    "0x354cca2f55dde182d36fe34d673430e226a3cb8c",  # XBridge
    "0x56ff4afd909aa66a1530fe69bf94c74e6d44500c",  # Tonken (APEMAGA)
    "0x2409af0251dcb89ee3dee572629291f9b087c668",  # InitializableImmutableAdminUpgradeabilityProxy (UwuLend)
    "0x02e7b8511831b1b02d9018215a0f8f500ea5c6b3",  # ParaSwapRepayAdapter (AAVE)
    "0x09a80172ed7335660327cd664876b5df6fe06108",  # OMPxContract
    "0x1bbf25e71ec48b84d773809b4ba55b6f4be946fb",  # VOWToken
    "0xe3a0bc3483ae5a04db7ef2954315133a6f7d228e",  # YodlRouter
    "0x047d41f2544b7f63a8e991af2068a363d210d6da",  # TransparentUpgradeableProxy (Bedrock_DeFi)
    "0x702696b2aa47fd1d4feaaf03ce273009dc47d901",  # Vault (Bedrock_DeFi)
    "0x240cd7b53d364a208ed41f8ced4965d11f571b7a",  # DOGGO
    "0xb3912b20b3abc78c15e85e13ec0bf334fbb924f7",  # HANA
    "0xf2c8e860ca12cde3f3195423ecf54427a4f30916",  # OTSeaStaking
    "0xf10bc5be84640236c71173d1809038af4ee19002",  # NFTLiquidation (OnyxDAO)
    "0xe0c218e1633a5c76d57ff4f11149f07bfff16aea",  # PLNTOKEN
    "0xe2910b29252f97bb6f3cc5e66bfa0551821c7461",  # PythiaTokenStaking
    "0x18775475f50557b96c63e8bbf7d75bfeb412082d",  # FireToken
    "0xbbbbbbbbbb9cc5e90e3b3af64bdaf62c37eeffcb",  # Morpho (MorphoBlue)
    "0x050163597d9905ba66400f7b3ca8f2ef23df702d",  # ChiSale
    "0x9008d19f58aabd9ed0d60971565aa8510560ab41",  # GPv2Settlement (CoW)
    "0x280a8955a11fcd81d72ba1f99d265a48ce39ac2e",  # VirtualToken (vETH)
    "0x1e791527aea32cddbd7ceb7f04612db536816545",  # Action (CGT)
    "0x4095f064b8d3c3548a3bebfd0bbfd04750e30077",  # EthereumBundlerV2 (MorphoBlue)

    # --- 2025 ---
    "0x439cac149b935ae1d726569800972e1669d17094",  # IdolMain (IdolsNFT)
    "0x05641e33fd15baf819729df55500b07b82eb8e89",  # PumpToken (LAURA)
    "0x4e34dd25dbd367b1bf82e1b5527dbbe799fad0d0",  # UnilendV2Pool (Unilend)
    "0x1db92e2eebc8e0c075a02bea49a2935bcd2dfcf4",  # Proxy (Bybit)
    "0x34cfac646f301356faa8b21e94227e3583fe3f5f",  # GnosisSafe (Bybit)
    "0x7094e706e75e13d1e0ea237f71a7c4511e9d270b",  # HegicPUT (HegicOptions)
    "0xf3f84ce038442ae4c4dcb6a8ca8bacd7f28c9bde",  # SilicaPools (Alkimiya_io)
    "0xb91ae2c8365fd45030aba84a4666c4db074e53e7",  # Vault (LeverageSIR)
    "0xa88800cd213da5ae406ce248380802bd53b47647",  # Settlement (OneInchFusionV1)
    "0x76ea342bc038d665e8a116392c82552d2605eda1",  # UniswapV2Pair (UNI)
    "0x934cbbe5377358e6712b5f041d90313d935c501c",  # Laundromat
    "0x35d8949372d46b7a3d5a56006ae77b215fc69bc0",  # TransparentUpgradeableProxy (UsualMoney)
    "0x37ea5f691bce8459c66ffceeb9cf34ffa32fdadc",  # GradientMarketMakerPool
    "0x48afbbd342f64ef8a9ab1c143719b63c2ad81710",  # TransparentUpgradeableProxy (MetaPool)
    "0x6e90c85a495d54c6d7e1f3400fef1f6e59f86bd6",  # ResupplyPair (ResupplyFi)
    "0x245a551ee0f55005e510b239c917fa34b41b3461",  # Staking (SWAPPStaking)
    "0xffb512b9176d527c5d32189c3e310ed4ab2bb9ec",  # RareStakingV1 (SuperRare)
    "0x54cd23460df45559fd5feeaada7ba25f89c13525",  # ERC1967Proxy
    "0xf4a21ac7e51d17a0e1c8b59f7a98bb7a97806f14",  # LeverageUp (SizeCredit)
    "0x46f54d434063e5f1a2b2cc6d9aaa657b1b9ff82c",  # PrivilegedCheckpointCauldronV4 (MIMSpell3)
    "0x6a06707ab339bee00c6663db17ddb422301ff5e8",  # DRLVaultV3

    # --- 2026 ---
    "0x764c64b2a09b09acb100b80d8c505aa6a0302ef2",  # AdminUpgradeabilityProxy (Truebit)
    "0x4822d9172e5b76b9db37b75f5552f9988f98a888",  # AdminUpgradeabilityProxy (AlkemiEarn)
]
