# 📈 Algorithmic Trading Agent using Deep Reinforcement Learning (PPO)

This project implements a Deep Reinforcement Learning (DRL) agent trained to perform algorithmic trading on the Borsa Istanbul, specifically focusing on the THYAO.IS stock. The core objective is to move beyond simple price prediction and develop an autonomous agent that learns an optimal trading policy (Buy, Sell, Hold) by maximizing a defined objective function based on the current market state.

The agent leverages Proximal Policy Optimization (PPO) from the `stable-baselines3` library and interacts with a custom trading environment built using `gymnasium` (formerly OpenAI Gym).

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/) [![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-lightgrey?logo=openai&logoColor=blue)](https://stable-baselines3.readthedocs.io/) [![Gymnasium](https://img.shields.io/badge/Gymnasium-gymnasium?color=green&logo=gymnasium)](https://gymnasium.farama.org/)

## 🎯 Project Goal

The primary goal is **not** to forecast exact price movements but to train an agent capable of making **intelligent, sequential decisions** under uncertainty to optimize cumulative returns or risk-adjusted returns (e.g., Sharpe Ratio) over time.

## ✨ Key Features & Technologies

* **DRL Algorithm:** Utilizes **Proximal Policy Optimization (PPO)**, a state-of-the-art reinforcement learning algorithm, via the `stable-baselines3` library.
* **Custom Trading Environment:** A flexible `gymnasium`-compliant environment (`TicaretOrtami`) simulating stock trading dynamics, including:
    * Transaction costs.
    * State representation with normalization (`MinMaxScaler`).
    * Configurable reward functions (e.g., simple PnL, change in Sharpe Ratio).
* **Feature Engineering:** Enriches the market state with:
    * **Technical Indicators:** Simple Moving Average (SMA) and Relative Strength Index (RSI) calculated using `pandas`.
    * **News Sentiment Analysis:** Daily sentiment scores derived from Google News headlines related to the target company ("THY"), processed using a pre-trained Turkish BERT model (`savasy/bert-base-turkish-sentiment-cased`) via the `transformers` library.
    * **Market Context:** Integration of broader market data (e.g., BIST100 index daily returns).
* **Data Handling:** Uses `yfinance` for downloading historical stock and index data, and `pandas` for manipulation and feature calculation.
* **Hyperparameter Optimization:** Leverages `Optuna` for systematic tuning of key PPO hyperparameters (learning rate, network architecture, etc.) to find the most effective agent configuration.
* **Backtesting Framework:** Includes scripts to evaluate the trained agent's performance on out-of-sample data (data not seen during training) and compare it against a simple "Buy and Hold" benchmark.
* **Environment Management:** Uses `venv` for managing project dependencies and ensuring reproducibility.

## 🛠️ Installation

1.  **Clone the Repository:**
    ```bash
    git clone (https://github.com/AlaattinUysal/OptimalTicaretSitesi.git)
    cd OptimalTicaretSitesi
    ```
2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Windows
    # source venv/bin/activate # Linux/macOS
    ```
3.  **Install Requirements:**
    * Create a `requirements.txt` file with the following content:
        ```text
        pandas
        numpy
        yfinance
        gymnasium
        stable-baselines3[extra]
        torch
        tensorflow # Primarily for transformer's backend if needed, adjust if using torch exclusively
        tf-keras # Compatibility layer for transformers
        requests
        beautifulsoup4
        transformers
        scikit-learn
        matplotlib
        optuna
        tqdm
        # sqlite3 comes with Python usually
        ```
    * Install the packages:
        ```bash
        pip install -r requirements.txt
        ```

## ⚙️ Usage

*(Ensure the virtual environment is activated before running any script)*

1.  **(Optional but Recommended) Pre-calculate Sentiment Scores:**
    This script fetches news and calculates sentiment scores for the entire training period. It saves the results to `thy_duygu_skorlari.csv` and supports resuming if interrupted. **Run this only once; it takes a long time.**
    ```bash
    python duygulari_on_hesapla.py
    ```

2.  **(Optional) Find Optimal Hyperparameters:**
    Run `Optuna` to search for the best PPO parameters. Results are saved in `optuna_study.db`. This also takes a significant amount of time.
    ```bash
    python optuna_optimize.py
    ```

3.  **Train the Optimized Agent:**
    Uses the best parameters found by Optuna (hardcoded in the script after optimization) to train the final agent and saves it as `ppo_champion_model.zip`. Adjust `total_timesteps` as needed.
    ```bash
    python ppo_egitim.py
    ```

4.  **Test the Trained Agent:**
    Loads the trained model (`ppo_champion_model.zip` by default) and evaluates its performance on unseen future data. Compares results against the "Buy and Hold" benchmark and plots the trades.
    ```bash
    python final_champion_test.py
    ```

## 📊 Findings & Key Takeaways

* The PPO agent successfully learned to interact with the environment and generate trading signals.
* Incorporating news sentiment provided a slight improvement in performance compared to models using only price-based indicators in initial trials.
* Systematic hyperparameter optimization using Optuna identified configurations yielding positive returns on the test set.
* Despite optimization and feature enrichment (including sentiment and market index data), the final agent, while profitable (+X.XX% on test data - *replace with your final percentage*), **did not consistently outperform the simple "Buy and Hold" benchmark** during the specific out-of-sample test period.
* This outcome underscores the inherent difficulty in achieving consistent alpha in financial markets using DRL with standard, publicly available data sources. The agent often converged towards conservative strategies (minimal trading) when faced with market uncertainty, especially with more complex state representations or reward functions (like Sharpe Ratio optimization).

## 🚀 Potential Future Work

* **Advanced Feature Engineering:** Incorporate volatility indicators (e.g., Bollinger Bands, ATR), macroeconomic data (interest rates, inflation), or alternative data sources (social media trends).
* **More Sophisticated Architectures:** Explore Attention mechanisms (Transformers) within the policy network to better capture long-range dependencies in market data.
* **Refined Reward Shaping:** Design more nuanced reward functions that explicitly encourage desired behaviors like risk management or capturing specific market patterns.
* **Portfolio Optimization:** Extend the framework to manage multiple assets simultaneously, optimizing capital allocation across a portfolio.
* **Different Algorithms:** Experiment with other DRL algorithms like SAC (Soft Actor-Critic) or A2C/A3C.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
