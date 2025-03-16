from setuptools import setup, find_packages
from osmosis_wrap.consts import package_name, package_version

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=package_name,
    version=package_version,
    author="Gulp AI",
    author_email="jake@gulp.ai",
    description="Monkey patches LLM client libraries to print all prompts and responses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Gulp-AI/osmosis-wrap",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "python-dotenv>=0.19.0",
        "requests>=2.25.0",
    ],
    extras_require={
        "anthropic": ["anthropic>=0.5.0"],
        "openai": ["openai>=0.27.0"],
        "langchain": ["langchain>=0.0.200"],
        "all": ["anthropic>=0.5.0", "openai>=0.27.0", "langchain>=0.0.200"],
    },
) 