from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(filepath : str) -> List[str]:
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

AUTHOR_USER_NAME   = 'AarizZafar'
REPO_NAME          = 'LLM_PDF_Q-A_QUERY'

setup(
    name                         = REPO_NAME,
    version                      = '0.0.1',
    author                       = AUTHOR_USER_NAME,
    author_mail                  = 'aariz.zafar01@gmail.com',
    packages                     = find_packages(),
    url                          = {f'https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}'},
    install_requirements         = get_requirements('requirements.txt')
)

