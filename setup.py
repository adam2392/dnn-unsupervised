from distutils.core import setup

setup(
    name='DNNTVB',
    version='1.1',
    packages=['eegdnn'],
    install_requires=['numpy', 'scipy', 
    		'scikit-learn', 'tensorflow', 'keras',
    		'pandas', 'tqdm', 'onnx', 'opencv-python'],
    url='',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Adam Li',
    description='deep learning from TVB simulated data.'
)
