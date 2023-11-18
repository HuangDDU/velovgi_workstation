"""
VeloVGI
see: https://github.com/HuangDDU/velovgi_workstation
"""

from setuptools import setup

setup(
        name="VeloVGI",
        version="0.0.2",
        description="Graph Variational Autoencoder for scRNA-seq velocity",
        author="HuangDDU",
        author_email="hzy554598474@163.com",
        url="https://github.com/HuangDDU/velovgi_workstation",
        packages=["velovgi"],
        install_requires=[
            "pot==0.9.0",
            "pytorch-lightning==1.7.7",
            "torch==1.12.1",
            "torch-geometric==2.0.1",
            "scvi-tools==0.19.0",
            "scanpy==1.9.3",
            "scvi-tools==0.19.0",
            "scvelo==0.2.5",
            "velovi==0.2.0"
        ]
)