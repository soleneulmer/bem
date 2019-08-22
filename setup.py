from setuptools import setup
import os, glob

here = os.path.abspath(os.path.dirname(__file__))
output = os.path.join(here, 'published_output/')

setup(name='bem',
      version='0.1.6',
      description='Random forest for exoplanets',
      url='https://github.com/soleneulmer/bem',
      author='Solène Ulmer-Moll',
      author_email='solene.ulmer-moll@astro.up.pt',
      license='MIT',
      packages=['bem'],
      install_requires=['numpy', 'scipy',
                        'pandas', 'scikit-learn',
                        'matplotlib', 'astropy',
                        'lime'],
      data_files=[('published_output', glob.glob('published_output/*'))],
      zip_safe=False,
)
