from setuptools import find_packages, setup

setup(
    name="wsdeval",
    version="0.0.1",
    description="WSD system evaluation Finnish",
    url="https://github.com/frankier/finn-wsd-eval/",
    author="Frankie Robertson",
    author_email="frankie@robertson.name",
    license="Apache v2",
    packages=find_packages(exclude=("tests", "scripts")),
    zip_safe=False,
)
