# Tweet Analysis Web App

This web application allows users to retrieve and analyze tweets based on specific search criteria such as username, hashtag, or term. The app uses the `ntscraper` model to perform analysis and visualizes the results using various plots and word clouds.

## Features

- Retrieve tweets using `Nitter` based on username, hashtag, or term.
- Translate and clean tweets.
- Detect the language of the tweets.
- Analyze the subjectivity and polarity of the tweets.
- Predict sentiment using a pre-trained model.
- Visualize results with word clouds, bar charts, and donut charts.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/tweet-analysis-web-app.git
    cd tweet-analysis-web-app
    ```

2. Create a virtual environment and activate it:

    ```bash
    python -m venv .venv
    .venv\Scripts\activate  # On Windows
    source .venv/bin/activate  # On macOS/Linux
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Make sure you have the necessary files and directories:
    - `twitter (2).png`: Twitter logo to be displayed in the app.
    - `sl_z_072523_61700_01.jpg`: Additional image to be displayed in the app.
    - `Nitter_logo.png`: Logo to be displayed in the sidebar.
    - `clf.pkl`: Pre-trained model file used for sentiment prediction.

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
