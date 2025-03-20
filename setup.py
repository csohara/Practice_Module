from setuptools import setup, find_packages

setup(
    name="Autograd",                     
    version="0.1.0",                       
    description="Autograd Attempt",        
    author="Caitlin O'Hara",                     
    packages=find_packages(where="src"),    
    package_dir={"": "src"},                
    install_requires=["torch", "numpy"],   
)
