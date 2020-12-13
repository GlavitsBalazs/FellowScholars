#https://github.com/jupyter/docker-stacks/tree/master/scipy-notebook

FROM jupyter/scipy-notebook:42f4c82a07ff

USER root

COPY install-cuda.sh /usr/local/bin/
RUN install-cuda.sh

USER $NB_UID

# Install tensorflow and librosa.
RUN pip install --quiet --no-cache-dir \
    'tensorflow==2.4.0rc2' 'librosa' && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"