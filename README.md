# U.S. Aviation Performance & Forecasting Analytics (PS-3)

Welcome to the **U.S. Aviation Performance & Forecasting Project**! This project was designed from the ground up to be **simple, friendly, and easy to run** for first-time data science students and users. 

Instead of dealing with complex web-servers or advanced Javascript frameworks, this project is built entirely in **Python** for the machine learning logic, and outputs a simple, gorgeous **HTML Dashboard** that you can open **just by double-clicking it in your browser**!

---

## 📂 File Structure

Here is a quick map of the files in your project folder:

```text
us_aviation_project/
├── data/
│   └── aviation_data.csv        # The dataset containing passenger and flight metrics
├── plots/                       # Folder where Python saves the output charts
│   ├── seasonality.png          # Seasonality analysis plot
│   ├── correlation.png          # Metric correlations heatmap
│   ├── confusion_matrix.png     # Performance classification confusion matrix
│   └── demand_forecast.png      # Forecasted passenger demand chart
├── generate_data.py             # Python script that generates the aviation dataset
├── ml_analysis.py               # The main machine learning script (Calculates all metrics & saves plots)
├── report.html                  # Beautiful static report viewer (Double-click to open!)
├── styles.css                   # Premium CSS styles (glassmorphism dark mode)
└── README.md                    # This friendly guide!
```

---

## 🚀 How to Run the Project (Step-by-Step)

To run this project, make sure you have Python installed on your computer.

### Step 1: Open a terminal in this folder
Open your terminal (PowerShell on Windows, or Terminal on macOS/Linux) and navigate to this folder.

### Step 2: Install required libraries
Run the following command to install the standard, reliable data science libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels
```

### Step 3: Generate the aviation dataset
Run the data generator script. This will create a highly realistic aviation dataset containing October 2002 to March 2017 metrics:
```bash
python generate_data.py
```
*You will see a message: `Dataset generated successfully! Saved 2784 rows to 'data/aviation_data.csv'.`*

### Step 4: Run the Machine Learning Models
Now run the core analysis script. This script will train the models, compute the metrics, save the plots into the `plots/` folder, and automatically compile your static report:
```bash
python ml_analysis.py
```
*It will print the Decision Tree and Holt-Winters metrics directly to your terminal!*

### Step 5: Open your Dashboard Report!
Navigate to your project folder using your operating system's file explorer and **double-click the `report.html` file**. It will open instantly in your web browser (Chrome, Edge, Safari, Firefox), displaying your results in a breathtaking, premium dark-mode dashboard. **No server setup required!**

---

## 🧠 Key Machine Learning Concepts Explained

Here are simple, intuitive explanations of the terms in this project, perfect for when you need to explain or present your work:

### 1. Classification Metrics (How well we identify underperformance)
Our classifier separates monthly operations into two bins: **Efficient** (Load Factor &ge; 78%) or **Underperforming** (Load Factor < 78%).
- **Accuracy**: The overall percentage of times the model predicted correctly. *Example: 85% means our model got 85 out of 100 months correct.*
- **Precision**: Out of all the months the model *predicted* as efficient, what percentage actually were. *(Protects against false positives).*
- **Recall**: Out of all the *actual* efficient months, what percentage did the model manage to find. *(Protects against false negatives).*
- **F1-Score**: A single number that balances both Precision and Recall. Closer to 1 is better.
- **Confusion Matrix**: A 2x2 grid showing exactly where the model succeeded and where it got confused (True Positives, True Negatives, False Positives, False Negatives).

### 2. Time-Series Forecasting (Predicting future demand)
Our forecasting model uses **Holt-Winters Triple Exponential Smoothing**, a classical statistical method that handles:
1. **Level**: The baseline traffic.
2. **Trend**: The general upward growth over the years.
3. **Seasonality**: The repeating yearly cycle (summer peaks, winter troughs).

### 3. Forecasting Metrics (How close were our predictions)
- **MAE (Mean Absolute Error)**: The average difference between our predicted passenger count and the actual count. Lower is better.
- **RMSE (Root Mean Squared Error)**: Similar to MAE, but it penalizes larger mistakes more heavily. Lower is better.
- **MAPE (Mean Absolute Percentage Error)**: The average percentage mistake we made. *Example: A MAPE of 3.2% means our predictions were, on average, within 96.8% of the real numbers!*

---

## 💡 Mitigation Strategies Used
When an airline or airport falls below the 78% efficiency limit, we implement two main strategies:
1. **Seasonal Capacity Realignment**: Trimming scheduled flights by 10-15% in low months (Jan/Feb) to prevent flying empty planes.
2. **Dynamic Leisure Pricing**: Running fare promotions during winter to attract price-sensitive travelers and fill seats.
