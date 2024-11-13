import streamlit as st
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import re

class PaperDatabase:
    def __init__(self):
        self.papers_file = Path("papers_database2.json")
        self.load_database()

    def load_database(self):
        if self.papers_file.exists():
            with open(self.papers_file, 'r') as f:
                self.papers = json.load(f)
        else:
            # Initialize with some sample deep learning papers
            self.papers = {
                "paper_a": {
                    "title": "Advances in Deep Learning for Computer Vision",
                    "year": 2023,
                    "authors": ["Smith, J.", "Johnson, M."],
                    "abstract": "This paper presents novel approaches to deep learning for object detection and image classification...",
                    "sections": {
                        "4.1": "Convolutional Neural Network Architectures",
                        "5.0": "Future Research Directions"
                    },
                    "keywords": ["computer vision", "deep learning", "CNN"]
                },
                "paper_b": {
                    "title": "Transformer-based Language Models for Natural Language Processing",
                    "year": 2022,
                    "authors": ["Brown, R.", "Davis, K."],
                    "abstract": "We propose new techniques for training large-scale transformer models for NLP tasks...",
                    "sections": {
                        "2.2": "Self-Attention Mechanism",
                        "3.5": "Model Scaling and Efficiency"
                    },
                    "keywords": ["NLP", "transformers", "language models"]
                },
                "paper_c": {
                    "title": "Generative Adversarial Networks for Image Synthesis",
                    "year": 2021,
                    "authors": ["Chen, L.", "Wang, X."],
                    "abstract": "This paper explores the use of generative adversarial networks (GANs) for generating realistic images...",
                    "sections": {
                        "2.1": "GAN Architecture",
                        "3.3": "Stabilizing GAN Training"
                    },
                    "keywords": ["GANs", "image synthesis", "generative models"]
                },
                "paper_d": {
                    "title": "Deep Reinforcement Learning for Game-Playing Agents",
                    "year": 2020,
                    "authors": ["Lee, M.", "Kim, J."],
                    "abstract": "We present a deep reinforcement learning approach for training agents to play complex games...",
                    "sections": {
                        "4.2": "Q-learning with Deep Neural Networks",
                        "5.1": "Exploration vs. Exploitation"
                    },
                    "keywords": ["reinforcement learning", "game AI", "deep Q-networks"]
                },
                "paper_e": {
                    "title": "Deep Learning for Time Series Forecasting",
                    "year": 2019,
                    "authors": ["Wang, Z.", "Li, T."],
                    "abstract": "This paper investigates the use of deep learning techniques for forecasting time series data...",
                    "sections": {
                        "3.1": "Recurrent Neural Networks",
                        "4.3": "Forecasting Accuracy Metrics"
                    },
                    "keywords": ["time series", "forecasting", "recurrent neural networks"]
                },
                "paper_f": {
                    "title": "Unsupervised Representation Learning with Deep Autoencoders",
                    "year": 2018,
                    "authors": ["Zhao, Y.", "Chen, X."],
                    "abstract": "We explore the use of deep autoencoders for learning effective representations from unlabeled data...",
                    "sections": {
                        "2.4": "Encoder-Decoder Architecture",
                        "4.1": "Dimensionality Reduction"
                    },
                    "keywords": ["unsupervised learning", "autoencoders", "representation learning"]
                }
            }
            self.save_database()

    def save_database(self):
        with open(self.papers_file, 'w') as f:
            json.dump(self.papers, f, indent=2)

    def search_papers(self, query, start_year=2018):
        results = []
        query = query.lower()
        current_year = datetime.now().year

        for paper_id, paper in self.papers.items():
            if paper['year'] >= start_year and paper['year'] <= current_year:
                if (query in paper['title'].lower() or
                    query in paper['abstract'].lower() or
                    any(query in keyword.lower() for keyword in paper['keywords'])):
                    results.append(paper)
        
        return results

    def get_paper_by_id(self, paper_id):
        return self.papers.get(paper_id)

class ResearchChatbot:
    def __init__(self):
        self.db = PaperDatabase()
        
    def process_query(self, query):
      query = query.lower()

    # Specific questions for each paper
      if "all the papers related to deep learning in the last 5 years" in query:
         return self.list_recent_papers()

      elif "summarize the advancements made in these papers" in query:
        return self.summarize_advances()

      elif "techniques are proposed in paper a" in query:
        return self.get_paper_techniques('a')

      elif "techniques are proposed in paper b" in query:
        return self.get_paper_techniques('b')
    
      elif "techniques are proposed in paper c" in query:
        return self.get_paper_techniques('c')
    
      elif "techniques are proposed in paper d" in query:
        return self.get_paper_techniques('d')
    
      elif "techniques are proposed in paper e" in query:
        return self.get_paper_techniques('e')
    
      elif "techniques are proposed in paper f" in query:
        return self.get_paper_techniques('f')
      elif "summarize the advancements made in these papers" in query:
          return self.summarize_advances()

      elif "techniques are proposed in paper b to improve language models" in query:
          return self.techniques_in_paper_b()
      
      elif "future research can be done based on these papers" in query:
          return self.suggest_future_research()
          
      # General pattern matching
      elif any(keyword in query for keyword in ['show', 'list', 'papers', 'recent']):
          return self.list_papers()
      
      elif 'a' in query:
          return self.summarize_advances()
      
      elif 'technique' in query or 'method' in query:
          match = re.search(r'paper ([a-zA-Z])', query)
          if match:
              return self.get_paper_techniques(match.group(1))
              
      elif 'future' in query and 'research' in query:
          return self.suggest_future_research()
          
      else:
          return "I can help you with:\n- Listing recent papers\n- Summarizing advances\n- Explaining techniques\n- Suggesting future research\n\nPlease rephrase your question."


    def list_recent_papers(self):
        papers = self.db.search_papers("", start_year=2018)
        
        if not papers:
            return "No papers found from 2018 to 2023."
        
        response = "Here are the papers published from 2018 to 2023:\n\n"
        for i, paper in enumerate(papers, 1):
            response += f"{i}. {paper['title']} ({paper['year']})\n"
            response += f"   Authors: {', '.join(paper['authors'])}\n"
            response += f"   Keywords: {', '.join(paper['keywords'])}\n\n"
        
        return response

    def techniques_in_paper_b(self):
        paper_b = self.db.get_paper_by_id("paper_b")
        if paper_b:
            return f"""Paper B proposes the following techniques to improve language models:
1. Self-Attention Mechanism
2. Model Scaling and Efficiency Improvements

Reference: Paper B - Section 2.2 and 3.5"""
        return "Paper B not found."

    def list_papers(self):
        papers = self.db.search_papers("", start_year=2018)
        response = "Recent papers on deep learning:\n\n"
        for i, paper in enumerate(papers, 1):
            response += f"{i}. {paper['title']} ({paper['year']})\n"
            response += f"   Authors: {', '.join(paper['authors'])}\n"
            response += f"   Keywords: {', '.join(paper['keywords'])}\n\n"
        return response

    def summarize_advances(self):
        return """Key advancements in deep learning (2018-2023):

1. Computer Vision:
   - Improved convolutional neural network architectures
   - Advances in object detection and image classification

2. Natural Language Processing:
   - Transformer-based language models
   - Improved model scaling and efficiency

3. Generative Models:
   - Advances in Generative Adversarial Networks (GANs)
   - Improved techniques for image synthesis

4. Reinforcement Learning:
   - Deep reinforcement learning for game-playing agents
   - Exploration-exploitation trade-offs

5. Time Series Forecasting:
   - Applying recurrent neural networks for forecasting
   - Improved forecasting accuracy metrics

6. Unsupervised Representation Learning:
   - Advances in deep autoencoder architectures
   - Effective dimensionality reduction techniques

Reference: Paper A - Section 4.1 | Paper B - Section 2.2 and 3.5"""

    def get_paper_techniques(self, paper_id):
        paper = self.db.get_paper_by_id(f"paper_{paper_id.lower()}")
        if paper:
            return f"""Techniques proposed in {paper['title']}:

1. {list(paper['sections'].values())[0]}
2. {list(paper['sections'].values())[1]}

Reference: Paper {paper_id.upper()} - Section {list(paper['sections'].keys())[0]}"""
        return "Paper not found. Please specify a valid paper identifier."

    def suggest_future_research(self):
        return """Potential future research directions in deep learning:

1. Scaling and Efficiency:
   - Improving the scalability of deep learning models
   - Reducing the computational and memory requirements

2. Interpretability and Explainability:
   - Enhancing the interpretability of deep learning models
   - Developing methods for explaining model decisions

3. Unsupervised and Self-Supervised Learning:
   - Advancing unsupervised representation learning techniques
   - Leveraging self-supervised learning for data-efficient training

4. Multimodal and Cross-Modal Learning:
   - Integrating and leveraging multiple data modalities
   - Developing models that can learn from and transfer across modalities

5. Safety and Robustness:
   - Improving the safety and robustness of deep learning models
   - Addressing vulnerabilities to adversarial attacks

Reference: Paper A - Section 5 | Paper D - Section 5.1"""

def main():
    st.set_page_config(
        page_title="Deep Learning Research Paper Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Deep Learning Research Paper Assistant")

    # Initialize chatbot and session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = ResearchChatbot()
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": """Welcome! I can help you explore research papers on deep learning."""}
        ]

    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # User input
    if prompt := st.chat_input("Ask about deep learning research..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        response = st.session_state.chatbot.process_query(prompt)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()