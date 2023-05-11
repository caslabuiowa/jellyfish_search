from glob import glob
import os
from setuptools import setup, find_packages

package_name = 'jellyfish_search'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files (note that they must have "launch" in the file name)
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
        # Include all config files
        (os.path.join('share', package_name, 'config'), glob('config/*.[jy][sma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='magicbycalvin',
    maintainer_email='magicbycalvin@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'jellyfish_controller = jellyfish_search.jellyfish_controller:main',
            'obstacle_detector = jellyfish_search.obstacle_detection_node:main'
        ],
    },
)
