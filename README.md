

# 🚀 Ocean Market Intelligence Engine

**By the Vibe Architect**

A decentralized, AI-driven market forecasting tool built on **Ocean Protocol**. This engine bridges the gap between raw web data and actionable market intelligence using decentralized compute.

---

## 🎯 Project Vision

In a crowded market, generic price trackers are insufficient. This project demonstrates a **Hybrid AI Engine** that performs three critical functions:

1. **Live Reality:** Connects to the **CoinGecko API** for real-time market data.
2. **Market Vibe:** Calculates **RSI** and **Volatility** to gauge market sentiment.
3. **Future Sight:** Utilizes a **Scikit-Learn Linear Regression** model to predict the next 24-hour price trend.

---

## 🛠️ The "Vibe Coding" Workflow

This project was developed using a **Vibe Coding** methodology—focusing on architectural intent and partnering with AI to curate high-performance code.

### 🧩 Key Technical Solves

* **Data Sanitization Layer:** Implemented a custom `dropna()` logic to handle API gaps, preventing dimension mismatch errors during AI training.
* **Decentralized Deployment:** Successfully deployed via the **Ocean Protocol VS Code Extension**, ensuring the code runs in a secure, private, and verifiable environment.
* **Artifact Generation:** Automated the creation of visual forecasts (`.png`) and technical reports (`.json`) within the Ocean `outputs` structure.

---

## 📂 Repository Structure

* `algo.py`: The core Python logic for data fetching, cleaning, and AI modeling.
* `requirements.txt`: Specified dependencies (`scikit-learn`, `requests`, etc.) for the Ocean Node.
* `Dockerfile`: Custom environment blueprint for consistent remote execution.
* `README.md`: Project documentation and vision.

---

## 🌊 How to Run on Ocean Protocol

1. **Setup:** Download and install VS Code then afterwards install the [Ocean Nodes VS Code Extension](https://www.google.com/search?q=https://marketplace.visualstudio.com/items%3FitemName%3DOceanProtocol.ocean-protocol).
2. **Initialize:** Open this folder in VS Code and select it as your "Project Folder" in the Ocean sidebar.
3. **Configure:** Click **Configure Compute** and select `algo.py` as your algorithm.
4. **Execute:** Click **Start FREE Compute Job**.
5. **Results:** Once the job reaches 100%, check your local `results` or job folder for the `ai_prediction_chart.png`.

---

## 📊 Sample Outputs

Upon a successful run, the engine delivers:

* **Market Report:** A JSON file containing the Volatility Index and RSI.
* **AI Forecast:** A visual chart showing actual prices vs. the AI's predicted trend line.

---

## 🏆 Conclusion

This repository serves as a blueprint for developers looking to merge **Web3 infrastructure** with **AI-driven market analysis**. It proves that with the right intent and decentralized tools, anyone can build a secure, real-world connected intelligence engine.


