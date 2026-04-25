# Bittensor and Selected Subnets — Brief Summary

## What is Bittensor?

Bittensor is a decentralized network built around open digital commodity markets, mostly for AI-related services. The blockchain layer is called **Subtensor**, and the network lets independent participants compete to produce useful outputs and get rewarded when the market judges that output as valuable. In simple terms, Bittensor is not one single AI product; it is a network of many specialized markets called **subnets**.

## What is a subnet?

A **subnet** is the main working unit inside Bittensor. Each subnet is its own incentive market for one specific kind of digital commodity. That commodity might be AI inference, GPU compute, weather forecasting, long-context optimization, 3D generation, lead generation, or something else.

Inside a subnet:

- **miners** do the useful work,
- **validators** test and score the miners,
- the subnet's incentive mechanism defines what “good work” means,
- and on-chain rewards are distributed based on validator scoring.

So a subnet is basically a competitive marketplace with its own task, rules, and ranking logic.

## What does Yuma Consensus do?

**Yuma Consensus** is the on-chain mechanism that converts validator scores into rewards.

Very simply:

1. Validators evaluate miners and submit weight scores.
2. The chain compares those scores across validators.
3. Scores that go too far beyond stake-backed network agreement get **clipped**.
4. Final miner and validator emissions are calculated from those consensus-clipped weights.

The main idea is that miners should earn more when multiple trusted validators independently agree they performed well, and validators should earn more when their judgments align with real consensus instead of noise, copying, or manipulation.

## How many subnets does Bittensor have?

Bittensor currently operates with a **128-subnet limit**. The Root network is **Subnet 0 (SN0)**, which is special and separate from normal application subnets. In practice, the network is commonly described as running at or near the full 128-slot limit, which is why subnet deregistration exists: to remove weak or inactive subnets and make room for new ones.

## Selected well-known subnets

Below is a very short summary of some famous subnets and what they are known for.

### SN64 — Chutes
A decentralized **serverless AI compute / inference** subnet. It lets developers deploy and run models without managing the underlying infrastructure, while miners supply the GPU execution layer.

### SN51 — Lium
A decentralized **GPU rental / compute marketplace**. Users rent GPU resources through the subnet, while miners contribute hardware capacity and are rewarded for useful compute.

### SN4 — Targon
Currently positioned as a **secure / confidential compute cloud** subnet. Public materials emphasize secure GPU and CPU rentals for model training and deployment.

### SN17 — 404-GEN
A decentralized **3D content generation** subnet. Miners compete to generate 3D assets or 3D outputs, and validators benchmark which systems perform best.

### SN18 — Zeus
A subnet focused on **environmental and weather forecasting**. It uses AI models to predict environmental variables from large climate-related datasets.

### SN24 — Quasar
A subnet focused on **long-context AI** and efficient attention systems. Current public repos describe miners competing to optimize flash-linear attention kernels and related performance.

### SN58 — Handshake
A subnet focused on an **AI provider marketplace and agent payments**. It aims to let agents discover providers, pay per request, and use Bittensor as the trust and scoring layer.

### SN71 — Leadpoet
A decentralized **AI sales lead generation / sales intelligence** subnet. It is aimed at finding, enriching, and scoring business leads.

### SN75 — Hippius

decentralized cloud storage for Bittensor. It focuses on persistent storage, with docs highlighting shard-based storage, erasure coding, and S3/IPFS-style usage so apps can store data in a decentralized way instead of relying only on normal cloud vendors.

### SN85 — Vidaio

decentralized video processing subnet. Its public materials focus on video upscaling and compression, with validators using quality metrics such as VMAF and related checks to score miner outputs.

### SN93 — Bitcast

decentralized creator marketing / influencer campaign subnet. Brands publish content briefs, creators make videos, and miners are rewarded based on measurable engagement signals such as watch time and analytics.


### SN42 — Gopher (ex-Masa)

decentralized data collection / data infrastructure subnet. It is positioned around gathering and normalizing AI-ready data from sources like web and social platforms, with downstream API/app usage built on top of that data layer.

### SN52 — Dojo

tied to trader tooling and market-facing Bittensor infrastructure through Tensorplex and Backprop. Public descriptions frame it as part of a trader-friendly ecosystem that helps users interact with Bittensor markets more effectively.

### SN8 — Vanta (formerly PTN / Proprietary Trading Network)

decentralized trading-signal / prop-trading subnet. Miners contribute trading models or signals, validators evaluate them, and the subnet tries to surface the best market intelligence.

### SN11 — TrajectoryRL

decentralized prompt optimization subnet. Miners compete to produce cheaper or better prompt/policy bundles for LLM tasks, and validators check correctness and cost efficiency.

### SN1 — Apex

one of the flagship agentic reasoning / frontier AI subnets. Its public positioning is around competitive open-source intelligence, advanced reasoning, and pushing model capability forward inside Bittensor.

### SN120 — Affine

subnet focused on cross-subnet coordination and reinforcement-learning-style improvement. Public writeups describe it as infrastructure that helps connect subnet capabilities and refine higher-level model performance.

## One-line view of the above subnets

- **Chutes, Lium, Targon** → compute infrastructure
- **404-GEN** → 3D generation
- **Zeus** → weather / environmental forecasting
- **Quasar** → long-context model optimization
- **Handshake** → agent marketplace + payments
- **Leadpoet** → AI sales leads

## Important note

Subnet positioning can change over time. Some teams evolve from one use case to another, or they tighten their public messaging after launch. So the descriptions above reflect current public docs, repos, and project pages rather than every historical version of those subnets.

## Sources

- [Bittensor Docs — Understanding Subnets](https://docs.learnbittensor.org/subnets/understanding-subnets)
- [Bittensor Docs — Yuma Consensus](https://docs.learnbittensor.org/learn/yuma-consensus)
- [Bittensor Docs — Subnet Deregistration](https://docs.learnbittensor.org/subnets/subnet-deregistration)
- [Bittensor Docs — btcli reference (Root / SN0 references)](https://docs.learnbittensor.org/btcli)
- [Chutes summary page](https://subnetalpha.ai/subnet/chutes/)
- [Lium docs](https://docs.lium.io/intro)
- [Targon website](https://targon.com/)
- [404-GEN repo](https://github.com/404-Repo/404-gen-subnet)
- [404-GEN website](https://www.404.xyz/)
- [Zeus repo](https://github.com/Orpheus-AI/Zeus)
- [Quasar repo](https://github.com/SILX-LABS/QUASAR-SUBNET/)
- [Handshake repo](https://github.com/Handshake58/HS58)
- [Handshake summary page](https://subnetalpha.ai/subnet/handshake/)
- [Leadpoet repo](https://github.com/leadpoet/leadpoet)
