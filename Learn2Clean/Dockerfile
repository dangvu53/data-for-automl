FROM python:3.7

WORKDIR /workspace

# Copy your project files into the container
COPY . /workspace

# Downgrade pip to a version compatible with legacy packages
RUN apt-get update && apt-get install -y build-essential gcc 
RUN pip install pip==20.3.4 setuptools==44.1.1 wheel==0.36.2
RUN pip install numpy==1.15.4
RUN pip install pandas==0.23.0 matplotlib==2.2.2 scipy==1.1.0 seaborn==0.8.1 joblib==0.13.1 statsmodels==0.9.0 sklearn==0.0 sklearn-contrib-py-earth==0.1.0 py_stringmatching==0.4.0 py_stringsimjoin==0.1.0 impyute==0.0.7 tdda==1.0.13 jellyfish==0.7.1
# Install Learn2Clean in development mode
RUN pip install -e /workspace/Learn2Clean/python-package

# Install Jupyter for notebook usage
RUN pip install notebook

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]