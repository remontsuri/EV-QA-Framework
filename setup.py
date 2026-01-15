from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ev-qa-framework',
    version='1.0.0',
    author='remontsuri',
    author_email='remontsuri@github.com',
    description='ML-powered QA Framework for Electric Vehicle & IoT Testing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/remontsuri/EV-QA-Framework',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Testing',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=[
        'pytest>=7.0',
        'pytest-cov>=4.0',
        'pytest-asyncio>=0.20',
        'scikit-learn>=1.0',
        'pandas>=1.5',
        'numpy>=1.20',
        'python-can>=4.0',
    ],
    extras_require={
        'dev': [
            'black>=22.0',
            'flake8>=4.0',
            'mypy>=0.990',
            'pylint>=2.14',
        ],
        'docs': [
            'sphinx>=5.0',
            'sphinx-rtd-theme>=1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'ev-qa=ev_qa_framework.cli:main',
        ],
    },
)
