from setuptools import setup, find_packages

print(find_packages('src'))

setup(
    name='aibenchmark',
    version='0.0.1',
    license='MIT',
    author="Based Labs",
    author_email='',
    packages=find_packages(exclude=["tests"]),
    url='https://github.com/BasedLabs/aibenchmark/',
    keywords='ai benchmark metrics',
    install_requires=[
        'scikit-learn',
        'torch',
        'py7zr',
        'gdown',
        'transformers'
        'pytest',
        'numpy',
        'pillow'
    ],
)
