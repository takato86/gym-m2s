from setuptools import setup

setup(name='gym_m2s',
      packages=["gym_m2s", "gym_m2s.robotics", "gym_m2s.robotics.fetch"],
      version='0.0.1',
      install_requires=['gym==0.21.0']  # And any other dependencies foo needs
)
