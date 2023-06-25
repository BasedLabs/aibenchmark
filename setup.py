from setuptools import setup, find_packages

print(find_packages('src'))

setup(
    name='aibench',
    version='0.0.2',
    long_description='Benchmark your model against other models',
    license='MIT',
    author="Based Labs",
    author_email='',
    packages=find_packages(exclude=["tests"]),
    url='https://github.com/BasedLabs/aibenchmark/',
    keywords='ai benchmark metrics',
    install_requires=[
        'datasets==2.13.0',
        'gdown==4.7.1',
        'lxml==4.9.2',
        'numpy==1.25.0',
        'pandas==2.0.2',
        'py7zr==0.20.5',
        'pytest==7.3.2',
        'Requests==2.31.0',
        'scikit_learn==1.2.2',
        'setuptools==68.0.0',
        'torch==2.0.1',
    ]
)
