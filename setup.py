from setuptools import setup, find_packages

setup(
    name="urbanmarl",
    version="0.1.0",
    description="Multi-Agent RL Benchmark for Urban UAV-MEC Networks",
    author="Basheer Raddwan",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "torchrl>=0.13.3",
        "tensordict>=0.3.0",
        "benchmarl @ git+https://github.com/yemenlinux/BenchMARL.git",
        "hydra-core>=1.3.2"
    ],
    python_requires=">=3.9",
)
