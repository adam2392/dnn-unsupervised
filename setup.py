from distutils.core import setup

setup(
    name='DNNTVB',
    version='1.1',
    packages=['eeglearn'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'tensorflow', 'keras'],
    url='',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Adam Li',
    description='Unsupervised deep learning from TVB simulated data.'
)
