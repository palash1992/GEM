try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages



VERSION = '2.0.1'
DISTNAME = 'nxt_gem'
MAINTAINER = 'Jan Ernsting'
MAINTAINER_EMAIL = 'j.ernsting@uni-muenster.de'
DESCRIPTION = 'nxt_gem: A Python module for Graph Embedding Methods'
LONG_DESCRIPTION = open('README.md').read()
URL = 'https://github.com/jernsting/nxt_gem'
DOWNLOAD_URL = 'https://github.com/jernsting/nxt_gem/archive/' + VERSION + '.tar.gz'
KEYWORDS = ['graph embedding', 'network analysis',
            'network embedding', 'data mining', 'machine learning']
LICENSE = 'BSD'
ISRELEASED = True

INSTALL_REQUIRES = (
    'numpy>=1.12.0',
    'scipy>=0.19.0',
    'networkx>=2.4',
    'matplotlib>=2.0.0',
    'scikit-learn>=0.21.2',
    'theano>=0.9.0',
)


def setup_package():
    setup(
        name=DISTNAME,
        version=VERSION,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        url=URL,
        download_url=DOWNLOAD_URL,
        keywords=KEYWORDS,
        install_requires=INSTALL_REQUIRES,
        packages=find_packages(),
        include_package_data=True,
        license=LICENSE,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        classifiers=['Intended Audience :: Science/Research',
                     'Intended Audience :: Developers',
                     'License :: OSI Approved',
                     'Programming Language :: Python',
                     'Topic :: Software Development',
                     'Topic :: Scientific/Engineering',
                     'Topic :: Scientific/Engineering :: Artificial Intelligence',
                     'Programming Language :: Python :: 3', ],
        )


if __name__ == "__main__":
    setup_package()
