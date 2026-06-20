# Dataset: Olivetti Faces

No download step is needed. The notebook calls `sklearn.datasets.fetch_olivetti_faces()`,
which caches the data locally on first run (about 7 MB).

- Source: AT&T Laboratories Cambridge, distributed with scikit-learn
- Contents: 400 greyscale images, 40 subjects, 10 images per subject, 64x64 pixels
- Access: public, included in scikit-learn
- Citation: Olivetti Research Laboratory, "The Database of Faces" (formerly ORL)

The task is closed-set identification: given a held-out image, return the subject. Images
of the same subject differ in expression, lighting, glasses and slight pose, which is what
makes the intra-class variation visible in `assets/intra-class-variation.png`.
