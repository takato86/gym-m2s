from setuptools import setup

setup(name='gym_m2s',
      packages=["gym_m2s", "gym_m2s.robotics"],
      version='0.0.1',
      install_requires=['gym', 'gym_robotics']  # And any other dependencies foo needs
)
