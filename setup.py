from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

        return requirements


setup(
    name='CementStrengthPrediction',
    version='0.0.1',
    author='Suzal',
    author_email='suzalcement@gmail.com',
    install_requires=get_requirements(r'C:\DataScience\Projects\cement_stregnth_prediction\requirements.txt'),
    packages=find_packages()

)