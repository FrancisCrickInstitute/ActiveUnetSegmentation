from setuptools import setup

setup(name='unetsl',
      version='0.1.0',
      packages=['unetsl', 'unetsl.scripts', 'unetsl.cerberus', "unetsl.masker", "unetsl.boxer"],
      #This requires tensorflow 2.x and numpy, these need to be installed independently.
      install_requires=["scikit-image==0.16.2", "urwid","click" ],
      entry_points={
          'console_scripts': [
              'create_model = unetsl.scripts.create_model:main',
              'attach_data_sources = unetsl.scripts.attach_data_source:main',
              'inspect_data_sources = unetsl.scripts.inspect_data_sources:main',
              'train_model = unetsl.scripts.train_model:main',
              'predict_image = unetsl.scripts.predict_image:main',
              'label_issues = unetsl.scripts.label_issues:main',
              'transfer_weights = unetsl.scripts.transfer_weights:main', 
              'cerberus = unetsl.cerberus.__main__:cerbs'
              
          ]
      },
      )
