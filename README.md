# RAG-Enhanced LLM Project

## Overview

While Large Language Models (LLMs) demonstrate impressive abilities, they grapple with issues such as hallucination, reliance on outdated information, and the utilization of opaque, untraceable reasoning processes. As a potential solution, Retrieval-Augmented Generation (RAG) has emerged, offering the integration of external database knowledge to address these challenges. This integration significantly boosts the precision and trustworthiness of the generated content, especially in tasks that require substantial knowledge. It also facilitates ongoing updates of knowledge and the incorporation of domain-specific information. RAG effectively combines the inherent knowledge of LLMs with the extensive and constantly evolving data in external databases.

## Project Goal

Our goal throughout this project is to harness the power of RAG in domains that garner significant interest from people, such as news and finance. By providing individuals with the ability to inquire about recently occurring events and receive well-updated responses, we aim to meet the growing demand for timely and accurate information. Achieving this goal necessitates the integration of a powerful generator, such as what RAG offers when combined with an LLM model.

Within this project, our aim is to introduce a ground-breaking RAG that promises to provide individuals with an unparalleled experience, allowing them to explore the latest updates from around the world through this innovative approach.

## Prerequisites

- Python 3.7 or higher
- Git
- Pip

## Installation

1. **Clone the Repository**

   ```sh
   git clone https://github.com/Rachid-Mek/RAG.git
   cd RAG
   ```

2. **Create a Virtual Environment**

   ```sh
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install Dependencies**

   ```sh
   pip install -r requirements.txt
   ```

## Usage

1. **Run the Application**

   ```sh
   python app.py
   ```

2. **Interact with the Model**

   Use the interface provided by the application to input your queries and receive responses generated through the RAG model.

## Contributing

We welcome contributions to this project. If you have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to thank the developers of LLAMA 3 and the RAG framework for their groundbreaking work which made this project possible.

---

title: Llama3-8b-RAG News Finance
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0

---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).
