# Game Recommendation System

A Streamlit application that recommends games based on a selected game and displays details and reviews with positive or negative titles depending on the review sentiment of the selected game.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Features
- **Game Recommendations**: Get game recommendations based on the selected game.
- **Game Details**: View detailed information about the recommended games.
- **Review Sentiment Analysis**: Reviews are categorized as positive or negative based on their content.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Game-Recommendation-System.git
    cd Game-Recommendation-System
    ```
    
2. Download the pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1AP-8RmNgTUGKMnBxud3uZMxczWQ8LHpF?usp=drive_link) and place it in the appropriate directory. Alternatively, you can create the model from scratch by following the steps in the [Usage](#usage) section.

## Usage

### Running the App with Pre-Trained Model

1. Make sure the pre-trained model is downloaded and placed in the appropriate directory.
2. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

### Creating the Model from Scratch

1. Run the scrapers to collect data:
    ```bash
    python RawgScraper.py
    python RawgReviewsScraper.py
    ```

2. Process the data and create the model by running the notebook:
    ```bash
    jupyter notebook Game_Recommender.ipynb
    ```

3. Once the model is created, you can run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add some feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or issues, please contact [Sahil](mailto:sahilkumarsingh8079@gmail.com).
