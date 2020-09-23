from setuptools import setup, find_packages

setup(
    name="nnetplot",
    version="0.0.1",
    description="Draw neural network architectures with matplotlib.",
    url="http://www.github.com/kristianuruplarsen/nnetplot",
    author='Kristian Urup Olesen Larsen',
        license='MIT',
        packages=find_packages(),
        install_requires=[
            'numpy',
            'matplotlib'
        ],    
)
