# 🩺 *PIE*-Med: *P*redicting, *I*nterpreting and *E*xplaining Medical Recommendations 

<div align="center">
    <a href="https://huggingface.co/spaces/praiselab-picuslab/PIE-Med" target="_blank"><img alt="PIE-Med"
        src="https://img.shields.io/badge/HuggingFace-PIE--Med-grey?style=for-the-badge&logo=huggingface&logoSize=auto&color=gold"/></a>
    <a href="LICENSE" target="_blank"><img alt="License"
        src="https://img.shields.io/badge/license-cc_by_nc_4.0-gray?style=for-the-badge&logo=creativecommons&logoColor=white&logoSize=auto&color=green"/></a>
</div>
<hr>

Welcome to the repository for **PIE-Med**, a cutting-edge system designed to enhance medical decision-making through the integration of Graph Neural Networks (GNNs), Explainable AI (XAI) techniques, and Large Language Models (LLMs).

## 📝 Citation

If you use **PIE-Med** in your research or find it helpful, please cite our work:

> Romano, A., Riccio, G., Postiglione, M., & Moscato, V. (2024). *PIE-Med: Predicting, Interpreting and Explaining Medical Recommendations* [https://doi.org/10.1007/978-3-031-88720-8_2](https://doi.org/10.1007/978-3-031-88720-8_2)


## 🎥 Demo (or GIF)
[Watch our demo](https://drive.google.com/file/d/1e9VXslnBzOOp5QHh4GTrT-La1PdKxhzS/preview) to see PIE-Med in action and learn how it can transform healthcare recommendations!

## 📊 Data Source
We use the **[MIMIC-III](https://mimic.physionet.org/)** dataset, a freely accessible critical care database containing de-identified health information, including vital signs, laboratory test results, medications, and more. You can find more details about the dataset here:

## 🛠 Technologies Used
- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis ([Pandas Documentation](https://pandas.pydata.org/))
- **PyHealth**: Medical data preprocessing ([PyHealth Documentation](https://pyhealth.readthedocs.io/en/latest/))
- **PyTorch Geometric**: Building and training GNNs ([PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/))
- **Integrated Gradients & GNNExplainer**: Interpretability techniques ([PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/en/latest/))
- **Streamlit**: User interface development ([Streamlit Documentation](https://streamlit.io/))
- **Py AutoGen Multi-Agent Conversation Framework**: Multi-agent collaboration and explanation ([Py AutoGen Documentation](https://microsoft.github.io/autogen/))

The PIE-Med system's computational requirements depend on the configuration used. For resource-limited environments, the light configuration with an Intel i7 CPU and 16GB RAM offers a basic but functional setup, suitable for testing on small datasets. However, more demanding tasks, such as working with larger datasets or leveraging advanced machine learning techniques (e.g., Graph Neural Networks), benefit from cloud setups like the complete configuration, which includes a GPU (NVIDIA Tesla T4). In resource-constrained contexts, optimizing models and reducing dataset size would be crucial to ensure feasible performance.

## 🔬 Methodological Workflow
PIE-Med follows a comprehensive Predict→Interpret→Explain (PIE) paradigm:

1. **Prediction Phase**: We construct a heterogeneous patient graph from MIMIC-III data and apply GNNs to generate personalized medical recommendations.
2. **Interpretation Phase**: Integrated Gradients and GNNExplainer techniques are used to provide insights into the GNN's decision-making process.
3. **Explanation Phase**: A collaborative ensemble of LLM agents analyzes the model's outputs and generates comprehensive, understandable explanations.

![image](https://github.com/picuslab/PIE-Med/blob/main/PIE-Med.png)

## 🌟 Key Features
- **Integration of GNNs and LLMs**: Combining structured machine learning with natural language processing for robust recommendations.
- **Enhanced Interpretability**: Using XAI techniques to make the decision-making process transparent.
- **Collaborative Explanation**: Multi-agent LLMs provide detailed and understandable recommendations.

## 🚀 Getting Started
Follow these steps to set up and run PIE-Med on your local machine:

### Prerequisites
Ensure you have the following installed:
- Python 3.7+

### Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/picuslab/PIE-Med.git
    cd PIE-Med
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
1. **Run the Streamlit application**:
    ```bash
    streamlit run dashboard.py
    ```

   Open your web browser and go to `http://localhost:8501` to interact with the application.

## 📈 Conclusions
PIE-Med showcases the potential of combining GNNs, XAI, and LLMs to improve medical recommendations, enhancing both accuracy and interpretability. Our system effectively separates prediction from explanation, reducing biases and enhancing decision quality.

## ⚖ Ethical considerations

**PIE-Med** aims to support medical decision-making, but is not a substitute for professional medical advice. Users should confirm recommendations with authorised healthcare providers, as limitations of AI may affect accuracy. The system ensures transparency through interpretability techniques, but all results should be considered complementary to expert advice. **⚠️ Please note that the following repository is only a DEMO, with anonymised data used for illustrative purposes only**. 

### 🖋️ **Citation**

Please cite this work as follows:

```bibtex
@inproceedings{romano2025pie,
  title={PIE-Med: Predicting, Interpreting and Explaining Medical Recommendations},
  author={Romano, Antonio and Riccio, Giuseppe and Postiglione, Marco and Moscato, Vincenzo},
  booktitle={European Conference on Information Retrieval},
  pages={6--12},
  year={2025},
  organization={Springer}
}
```

## 🙏 Acknowledgments
We extend our gratitude to the creators of the MIMIC-III database, the developers of the Python libraries used, and our research team for their contributions to this project.

👨‍💻 This project was developed by Antonio Romano, Giuseppe Riccio, Marco Postiglione and Vincenzo Moscato

## 📜 License

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg
