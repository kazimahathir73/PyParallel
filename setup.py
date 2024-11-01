from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    
setup(
    name="pyparallel",
    version="0.1.0",
    author="Kazi Mahathir Rahman",
    author_email="mahathirmahim73@gmail.com",
    description="A PyTorch Deep Learning Framework for Easy Model Parallel Training and real-time Monitoring of CPU, GPU, and TPU. ",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kazimahathir73/PyParallel/",
    packages=find_packages(),
    install_requires=requirements,
    install_requires=[
        "dash>=2.0.0",
        "plotly>=5.0.0",
        "psutil>=6.1.0",
        "GPUtil>=1.4.0",
        "torch>=2.0.0",
        "torch-xla>=2.5.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)