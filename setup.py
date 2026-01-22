"""
Setup script for spiral-time-memory package

Install with: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ]
else:
    requirements = [
        'numpy>=1.24.0',
        'scipy>=1.11.0',
        'matplotlib>=3.7.0',
        'pytest>=7.4.0',
    ]

setup(
    name="spiral-time-memory",
    version="0.1.0",
    author="Marcel Krüger",
    author_email="marcelkrueger092@gmail.com",
    description="A falsifiable framework for non-Markovian quantum dynamics with temporal memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dfeen87/spiral-time-memory",
    
    packages=find_packages(exclude=["tests", "tests.*", "examples", "paper"]),
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.9",
    
    install_requires=requirements,
    
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'pytest-xdist>=3.3.0',
            'black>=23.11.0',
            'flake8>=6.1.0',
            'isort>=5.12.0',
            'mypy>=1.7.0',
            'pre-commit>=3.5.0',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'notebook>=7.0.0',
            'ipykernel>=6.25.0',
            'ipywidgets>=8.1.0',
            'seaborn>=0.12.0',
        ],
        'docs': [
            'sphinx>=7.2.0',
            'sphinx-rtd-theme>=2.0.0',
            'nbsphinx>=0.9.0',
        ],
        'quantum': [
            'qutip>=4.7.0',
        ],
        'all': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.11.0',
            'jupyter>=1.0.0',
            'sphinx>=7.2.0',
            'qutip>=4.7.0',
        ],
    },
    
    package_data={
        'theory': ['*.py'],
        'analysis': ['*.py'],
        'simulations': ['*.py'],
        'experiments': ['**/*.py'],
    },
    
    entry_points={
        'console_scripts': [
            'spiral-time-test=tests.run_all:main',
        ],
    },
    
    project_urls={
        "Bug Reports": "https://github.com/dfeen87/spiral-time-memory/issues",
        "Source": "https://github.com/dfeen87/spiral-time-memory",
        "Documentation": "https://github.com/dfeen87/spiral-time-memory/blob/main/THEORY.md",
        "Paper": "https://www.researchgate.net/publication/399958489_Spiral-Time_with_Memory_as_a_Fundamental_Principle_From_Non-Markovian_Dynamics_to_Measurement_without_Collapse?channel=doi&linkId=69714718e806a472e6a50958&showFulltext=true",
    },
    
    keywords=[
        'quantum mechanics',
        'non-markovian dynamics',
        'temporal memory',
        'measurement problem',
        'process tensor',
        'cp-divisibility',
        'falsification',
        'quantum foundations',
    ],
    
    zip_safe=False,
    
    # Additional metadata
    license="MIT",
    platforms=["any"],
)

# Post-installation message
print("""
╔════════════════════════════════════════════════════════════════════════╗
║                   Spiral-Time with Memory                              ║
║            A Falsifiable Framework for Non-Markovian QM                ║
╚════════════════════════════════════════════════════════════════════════╝

✓ Installation complete!

Quick Start:
  1. Read THEORY.md for theoretical background
  2. Read IMPLEMENTATIONS.md for code disclaimers
  3. Run tests: pytest tests/
  4. Explore: jupyter notebook examples/spiral_time_intro.ipynb

Important Reminders:
  • All implementations are illustrative, not canonical
  • Memory kernel forms are non-unique
  • Experimental validation is pending

Falsification Criterion:
  If all process tensors are CP-divisible under controlled
  interventions, the theory is FALSIFIED.

For support: See GitHub Issues or README.md
""")
