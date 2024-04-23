
# ComplAi Genie

## Overview

ComplAi Genie is an innovative SaaS application built to provide precise, accurate answers to regulatory queries, specifically focusing on RBI circulars. It leverages a combination of RAG + LLM and a streamlined user interface to help financial professionals easily navigate complex regulatory environments.

You can checkout a demo here: [Youtube](https://www.youtube.com/watch?si=fOyKV5tJ8YO6Ss61&v=rEp0MwHCUN8&feature=youtu.be)

## Features

- **Regulatory Query Tool**: Instantly provides answers and insights related to financial services and compliance queries, leveraging a vast database of regulatory documents.
- **Training Module**: Offers an educational interface for banking staff, significantly reducing the training period and facilitating rapid deployment in customer-facing roles.

## Target Group

- **Urban Cooperative Banks (UCBs)**
- **Non-Banking Financial Companies (NBFCs)**
- **FinTech Startups**
- **Retail Banking Staff**

## Usage

1. **Query Handling**: Simply type your regulatory or financial service-related question into the query box and receive precise, context-aware answers.
2. **Training Support**: Use the training module to access interactive learning resources designed specifically for new bank employees.

## Getting Started

### Prerequisites

- Python 3.8+
- Streamlit
- FAISS
- Sentence Transformers
- OpenAI API Key

### Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/complai-genie.git
cd complai-genie
```

Install the required packages:
```bash
pip install -r requirements.txt
```

### Configuration

Set up your OpenAI API key in your environment or `.streamlit/secrets.toml`:
```toml
[secrets]
OpenAI_key = "your_openai_api_key_here"
```

### Running the Application

Launch the app by running:
```bash
streamlit run app.py
```

Access the app through your web browser at the address shown in the terminal, typically `http://localhost:8501`.

## Contributing

We welcome contributions from the community, whether they are feature requests, improvements, or bug fixes. Please follow the standard fork-branch-pull request workflow.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Abhishek Mishra - [abhishekmishramm2@gmail.com](mailto:abhishekmishramm2@gmail.com)
Project Link: [hhttps://github.com/abhishekmishramm1997/complAi](https://github.com/abhishekmishramm1997/complAi)

