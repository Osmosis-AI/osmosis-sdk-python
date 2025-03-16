from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="osmosis-wrap",
    version="0.1.0",
    author="Osmosis AI",
    author_email="your.email@example.com",
    description="Monkey patches LLM client libraries to print all prompts and responses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/osmosis-wrap",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "python-dotenv>=0.19.0",
    ],
    extras_require={
        "anthropic": ["anthropic>=0.5.0"],
        "openai": ["openai>=0.27.0"],
        "all": ["anthropic>=0.5.0", "openai>=0.27.0"],
    },
) 