from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ev-qa-framework',
    version='1.0.0',
    author='remontsuri',
    author_email='remontsuri@github.com',
    description='ML-powered QA Framework for Electric Vehicle & IoT Battery Testing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/remontsuri/EV-QA-Framework',
    project_urls={
        'Bug Tracker': 'https://github.com/remontsuri/EV-QA-Framework/issues',
        'Documentation': 'https://github.com/remontsuri/EV-QA-Framework#readme',
        'Source Code': 'https://github.com/remontsuri/EV-QA-Framework',
        'Changelog': 'https://github.com/remontsuri/EV-QA-Framework/blob/main/CHANGELOG.md',
    },
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Automotive Industry',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Testing',
        'Topic :: Software Development :: Quality Assurance',
        'Topic :: System :: Networking :: Monitoring :: Hardware',
    ],
    keywords='ev, electric-vehicle, battery, bms, battery-management-system, '
             'anomaly-detection, quality-assurance, soh-prediction, '
             'battery-degradation, can-bus, telemetry, iot, automotive',
    python_requires='>=3.8',
    install_requires=[
        'pytest>=7.0',
        'pytest-cov>=4.0',
        'pytest-asyncio>=0.21',
        'scikit-learn>=1.2',
        'pandas>=2.0',
        'numpy>=1.24',
        'python-can>=4.3',
        'pydantic>=2.0',
        'fastapi>=0.100',
        'uvicorn>=0.23',
        'websockets>=11.0',
        'jinja2>=3.1',
        'pyyaml>=6.0',
        'aiohttp>=3.9',
        'aiofiles>=23.1',
    ],
    extras_require={
        'ml': ['tensorflow>=2.15'],
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
