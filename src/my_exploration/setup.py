from setuptools import find_packages, setup
from glob import glob

package_name = 'my_exploration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
                     glob('launch/*launch.[pxy][yma]*')),
        ('share/' + package_name + '/data',
                     glob('data/*.*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'project = my_exploration.project:main',
            'project_tf = my_exploration.project_tf:main',
            'project_yolo = my_exploration.project_yolo:main',
        ],
    },
)
