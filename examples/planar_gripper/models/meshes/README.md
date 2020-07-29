This folder contains the mesh files for the models.

For the simple brick model, separate visual geometries are used for illustration
and perception purposes. The `simple_brick.obj` is used only for perception.
Therefore, we rename the corresponding `.mtl` file since Drake will automatically
load the `.mtl` file (if there is a proper one) and use it for illustration.

In the future, if we want to use the texture for illustration, we only need to
rename the `.mtl` file to be the same name as the `.obj`.
