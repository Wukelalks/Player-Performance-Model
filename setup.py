from setuptools import setup, find_packages

setup(
     name='pperf',
     version='0.0.0',
     packages=find_packages(include=['*']),
     include_package_data=True,

     install_requires=[
         'numpy',
         'pandas',
         'scikit-learn',
         'streamlit',
     ],
)
