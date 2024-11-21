from setuptools import setup, find_packages

setup(
    name="hcap2",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "spacy",
        "nltk",
        "wordcloud",
        "matplotlib",
        "plotly",
        "altair",
    ],
) 