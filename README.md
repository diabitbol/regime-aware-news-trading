# Regime-Aware News Trading

A regime-aware commodity trading project combining **HMM-TVTP market regime detection**, **LLM-based extraction of commodity news shocks**, and a **Mixture of Experts** to model regime-dependent price reactions.

## Overview

This project studies whether the market impact of newly released commodity news depends on the prevailing market regime.

The main application is crude oil, with a focus on WTI or Brent. The core idea is to separate:

- the **market context before a news event**, modeled through latent regimes;
- the **new information contained in the news event**, extracted with NLP/LLM methods;
- the **subsequent market reaction**, modeled with supervised learning.

Rather than attempting to predict commodity returns purely from past market data, the project focuses on an event-driven question:

> **Does the same type of commodity news generate different price reactions depending on the market regime in which it occurs?**

## Project Architecture

The pipeline is organized into three main components.

### 1. Latent Market Regime Detection

A Hidden Markov Model with Time-Varying Transition Probabilities (**HMM-TVTP**) is used to identify latent commodity market regimes.

The regime itself is inferred from market variables such as:

- past oil returns;
- realized or rolling volatility;
- futures term structure;
- volume and open interest;
- potentially other market-based indicators.

The transition probabilities between regimes may additionally depend on slower-moving macroeconomic or fundamental variables, such as:

- inflation;
- interest rates;
- credit spreads;
- USD-related variables;
- oil inventories;
- aggregate past news sentiment.

The HMM-TVTP therefore estimates:

\[
P(S_t = k \mid \mathcal{F}_{t^-})
\]

where \(S_t\) is the latent market regime and \(\mathcal{F}_{t^-}\) represents the information available immediately before the arrival of the current news event.

Importantly, the **current news event is not used to define the regime**.

---

### 2. LLM-Based News Shock Extraction

Newly released commodity news is processed using NLP or LLM-based methods.

Instead of relying only on generic positive/negative sentiment, the goal is to extract economically meaningful information such as:

- shock type:
  - supply;
  - demand;
  - OPEC;
  - inventories;
  - geopolitical events;
- shock direction;
- shock intensity;
- uncertainty;
- potentially whether the information represents a surprise.

Each news event is therefore transformed into a structured feature vector:

\[
N_t =
[
\text{shock type},
\text{direction},
\text{intensity},
\text{uncertainty},
\ldots
]
\]

The objective is to represent the **new information arriving at time \(t\)** rather than information already embedded in historical prices.

---

### 3. Regime-Conditioned Mixture of Experts

The predictive component is formulated as a **Mixture of Experts**.

Each expert models how the market reacts to news under a particular latent regime:

\[
f_k(N_t) \rightarrow r_{t,t+h}
\]

where \(r_{t,t+h}\) is the oil return over a chosen post-news horizon.

The HMM regime probabilities act as soft gating weights:

\[
\hat r_{t,t+h}
=
\sum_{k=1}^{K}
P(S_t=k \mid \mathcal{F}_{t^-}) f_k(N_t)
\]

This allows the same news event to generate different expected reactions depending on the prevailing market environment.

For example, an unexpected OPEC production cut may have a stronger impact in a tight, backwardated oil market than in a weak-demand regime.

## Research Question

The central research question is:

> **Does conditioning the market response to commodity news on latent market regimes improve out-of-sample return prediction and trading performance?**

A secondary question is whether latent regime probabilities contain useful incremental information compared with using raw market variables directly.

## Benchmarks and Ablation Study

The regime-aware model will be compared against simpler alternatives such as:

- market-only model;
- news-only model;
- market + news model;
- news + hard regime assignment;
- news + soft HMM regime probabilities;
- full regime-conditioned Mixture of Experts.

This allows us to test whether the regime structure genuinely improves prediction rather than simply adding complexity.

## Prediction Targets

Several targets may be considered:

### Return regression

\[
y_t = r_{t,t+h}
\]

### Direction classification

\[
y_t = \mathbb{1}(r_{t,t+h} > 0)
\]

### Thresholded reaction classification

\[
y_t =
\begin{cases}
+1 & r_{t,t+h} > \tau \\
0 & |r_{t,t+h}| \leq \tau \\
-1 & r_{t,t+h} < -\tau
\end{cases}
\]

The appropriate horizon \(h\) will depend on the timestamps available for both news and market data.

## Backtesting

The evaluation will follow a strict out-of-sample procedure to avoid look-ahead bias.

Possible approaches include:

- expanding-window estimation;
- rolling-window estimation;
- filtered rather than smoothed HMM state probabilities;
- chronological train/validation/test splits.

At each prediction date, only information available at that time will be used.

Trading performance may be evaluated using:

- cumulative return;
- Sharpe ratio;
- maximum drawdown;
- hit rate;
- turnover;
- transaction costs;
- Calmar ratio.

## Why This Project?

Direct return prediction from historical market data is notoriously difficult because most observable information is already reflected in prices.

This project instead focuses on the interaction between:

1. **pre-existing market conditions**;
2. **newly arriving fundamental information**;
3. **subsequent market reaction**.

The regime model provides context, while the NLP component captures the new shock. The supervised model then learns whether the mapping from news to returns changes across regimes.

## Expected Pipeline

```text
Past market and macro data
            |
            v
        HMM-TVTP
            |
            v
  Regime probabilities
            |
            |----------------------+
            |                      |
Current commodity news             |
            |                      |
            v                      |
        NLP / LLM                  |
            |                      |
            v                      |
   News shock features             |
            |                      |
            +------> Mixture of Experts
                           |
                           v
                Predicted price reaction
                           |
                           v
                     Trading signal
